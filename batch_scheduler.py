"""Batch reply scheduling primitives for enhance mode.

This module intentionally depends only on the Python standard library so the
round state machine and the reply JSON handling can be unit tested without
importing AstrBot.

Design summary
--------------
* One :class:`BatchCoordinator` exists per ``unified_msg_origin``.
* The coordinator collects pending messages. A message event becomes the
  "leader" for its session and keeps its pipeline generator suspended until a
  round boundary (or an immediate wake request from an @/wake message). At that
  point the leader asks the coordinator for one :class:`BatchMessage` list and
  the caller issues a single ``event.request_llm(...)`` call.
* While a decision is in flight new messages only enter ``pending``; they are
  handled in a later round. ``@``/wake messages request an immediate flush that
  is honoured as soon as the in-flight call finishes (never concurrently).
* When ``silent_rounds`` consecutive rounds contain no new messages the session
  becomes ``SILENT`` and the leader releases its held event; a new message
  starts a fresh ACTIVE round.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

STATE_ACTIVE = "ACTIVE"
STATE_SILENT = "SILENT"

DISCORD_MAX_MESSAGE_LENGTH = 2000
DEFAULT_MAX_REPLIES = 5

# Event extra used to tell the LLM-request path which batch is being decided.
BATCH_PAYLOAD_EXTRA = "_enhance_batch_payload"

_JSON_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*$")
_MENTION_ID_RE = re.compile(r"^#?msg", re.IGNORECASE)


def normalize_message_id(raw: Any) -> str:
    """Normalize message ids so ``#msg123``/``msg123``/``123`` compare equal."""
    text = str(raw or "").strip().strip('"').strip("'")
    if text.startswith("#"):
        text = text[1:]
    text = _MENTION_ID_RE.sub("", text, count=1)
    if text.endswith(":"):
        text = text[:-1]
    return text.strip()


@dataclass(frozen=True)
class BatchMessage:
    """One collected user message that participates in a batch decision."""

    message_id: str
    user: str
    content: str
    time: str
    wake: bool = False

    def to_payload(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "user": self.user,
            "content": self.content,
            "time": self.time,
        }


@dataclass(frozen=True)
class ReplySpec:
    """One model-produced reply parsed from the batch JSON contract."""

    message_id: str
    text: str
    quote: bool


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _repair_json_control_chars(text: str) -> str:
    """Escape raw control characters inside JSON strings.

    Models occasionally emit literal newlines/tabs inside a JSON string, which
    is invalid JSON. Escaping them preserves the intended reply text instead of
    dropping the whole batch.
    """
    output: list[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
                output.append(char)
                continue
            if char == "\\":
                escaped = True
                output.append(char)
                continue
            if char == '"':
                in_string = False
                output.append(char)
                continue
            if char == "\n":
                output.append("\\n")
                continue
            if char == "\r":
                output.append("\\r")
                continue
            if char == "\t":
                output.append("\\t")
                continue
            output.append(char)
            continue
        if char == '"':
            in_string = True
        output.append(char)
    return "".join(output)


def _repair_trailing_commas(text: str) -> str:
    """Drop trailing commas before ``}``/``]`` outside of JSON strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            output.append(char)
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < length and text[lookahead].isspace():
                lookahead += 1
            if lookahead < length and text[lookahead] in "}]":
                index += 1  # skip the trailing comma
                continue
        output.append(char)
        index += 1
    return "".join(output)


def extract_json_object(raw_text: str | None) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    if not raw_text:
        return None
    stripped = _strip_code_fence(str(raw_text))
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = stripped[start : end + 1]
    attempts = (
        candidate,
        _repair_json_control_chars(candidate),
        _repair_trailing_commas(candidate),
        _repair_trailing_commas(_repair_json_control_chars(candidate)),
    )
    for attempt in attempts:
        try:
            parsed = json.loads(attempt)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def parse_batch_replies(
    raw_text: str | None,
    batch_message_ids: Iterable[str],
    max_replies: int = DEFAULT_MAX_REPLIES,
) -> list[ReplySpec]:
    """Parse the batch decision JSON into reply specs.

    Invalid/empty entries are skipped. ``message_id`` values outside of the
    current batch are kept as plain (non-quoted) replies so a reply is never
    silently dropped just because the model echoed a stale id.
    """
    if not raw_text:
        return []
    if str(raw_text).strip().startswith("<refuse/>"):
        return []

    payload = extract_json_object(raw_text)
    if payload is None:
        return []
    raw_replies = payload.get("replies")
    if not isinstance(raw_replies, list):
        return []

    valid_ids = {normalize_message_id(item) for item in batch_message_ids if item}
    limit = max(0, int(max_replies))
    replies: list[ReplySpec] = []
    for item in raw_replies:
        if len(replies) >= limit:
            break
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        message_id = normalize_message_id(item.get("message_id"))
        quote = _coerce_bool(item.get("quote"), True)
        if not message_id or (valid_ids and message_id not in valid_ids):
            message_id = message_id if message_id in valid_ids else ""
            quote = False
        replies.append(ReplySpec(message_id=message_id, text=text, quote=quote))
    return replies


def split_discord_text(
    text: str, limit: int = DISCORD_MAX_MESSAGE_LENGTH
) -> list[str]:
    """Split text into Discord-safe chunks, preferring newline boundaries.

    Newlines inside each chunk are preserved verbatim.
    """
    if not text:
        return []
    if limit <= 0 or len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for line in text.splitlines(keepends=True):
        while line:
            space = limit - len(current)
            if len(line) <= space:
                current += line
                line = ""
                continue
            if space > 0:
                current += line[:space]
                line = line[space:]
            if current:
                chunks.append(current)
                current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _truncate_history_text(history_text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(history_text) <= max_chars:
        return history_text
    return "...(older history truncated)...\n" + history_text[-max_chars:]


def build_batch_prompt(
    *,
    history_text: str,
    batch_payload: list[dict[str, str]],
    max_replies: int,
    mention_parse: bool,
    persona_name: str = "",
    persona_mask: str = "",
    max_history_chars: int = 0,
) -> str:
    """Build the single prompt used for one batch decision."""
    batch_json = json.dumps(batch_payload, ensure_ascii=False, indent=1)
    history = _truncate_history_text(history_text, max_history_chars)
    mention_rule = ""
    if mention_parse:
        mention_rule = (
            "- If you want to mention a user, use the control tag "
            '<mention id="USER_ID"/> inside the reply text. The user id can be '
            "found in the chat history after the nickname, e.g. "
            "[nickname/user_id/time]. Do NOT mention yourself.\n"
        )
    persona_block = ""
    if persona_mask:
        persona_block = (
            f"你当前的人格面具是：{persona_name or 'default'}\n"
            f"人格设定如下：\n{persona_mask}\n\n"
        )
    return (
        f"{persona_block}"
        "You are now in a chatroom. The chat history is as follows:\n"
        f"{history}\n\n"
        "New messages collected in the current round (JSON array):\n"
        f"{batch_json}\n\n"
        "Decide, in character, whether and how to respond to these new messages. "
        "You may reply to zero, one or several of them.\n"
        "Output ONLY a single JSON object (no markdown fences, no extra text):\n"
        '{"replies":[{"message_id":"<id of a new message>","text":"<reply text>","quote":true}]}\n\n'
        "Rules:\n"
        f"- Return between 0 and {max(0, int(max_replies))} replies. "
        'Return {"replies":[]} when you decide not to reply.\n'
        '- "message_id" MUST be copied from the "New messages" array above; never invent ids.\n'
        '- "text" may contain \\n line breaks; they are preserved when sent. '
        "Keep each reply under 2000 characters.\n"
        '- Set "quote": true to quote that message, or false for a plain reply.\n'
        "- Do not repeat the same reply text twice.\n"
        f"{mention_rule}"
        "- Use the SAME language as the chatroom.\n"
        '- Never output <refuse/>; just return {"replies":[]} to stay silent.\n'
        "- Only output your response and no other information."
    )


class BatchCoordinator:
    """ACTIVE/SILENT round state machine for one chat session."""

    def __init__(self, origin: str = "") -> None:
        self.origin = origin
        self.state = STATE_SILENT
        self.pending: deque[BatchMessage] = deque()
        self.empty_rounds = 0
        self.leader_active = False
        self.inflight = False
        self.immediate_flush = False
        self.closed = False
        self.decision_count = 0
        self.dropped_count = 0
        self._wake: asyncio.Event | None = None

    # ------------------------------------------------------------------
    # asyncio helpers
    # ------------------------------------------------------------------
    def _wake_event(self) -> asyncio.Event:
        if self._wake is None:
            self._wake = asyncio.Event()
        return self._wake

    def _request_wake(self) -> None:
        try:
            self._wake_event().set()
        except RuntimeError:  # pragma: no cover - no running loop
            pass

    # ------------------------------------------------------------------
    # Producer side (called by message handlers)
    # ------------------------------------------------------------------
    def submit(
        self,
        message: BatchMessage,
        *,
        wake: bool,
        max_pending: int,
    ) -> bool:
        """Enqueue one message and return True when the caller becomes leader."""
        if self.closed:
            return False

        self.pending.append(message)
        while len(self.pending) > max(1, int(max_pending)):
            self.pending.popleft()
            self.dropped_count += 1

        if wake:
            self.immediate_flush = True
            self._request_wake()

        if self.leader_active:
            return False

        self.leader_active = True
        self.state = STATE_ACTIVE
        self.empty_rounds = 0
        return True

    # ------------------------------------------------------------------
    # Leader side
    # ------------------------------------------------------------------
    def _take_pending(self) -> list[BatchMessage]:
        batch = list(self.pending)
        self.pending.clear()
        return batch

    def begin_inflight(self) -> None:
        self.inflight = True

    def end_inflight(self) -> None:
        self.inflight = False

    def end_leader(self) -> None:
        self.leader_active = False
        self.inflight = False
        if self._wake is not None:
            self._wake.clear()

    async def next_batch(
        self,
        *,
        round_seconds: float,
        silent_rounds: int,
    ) -> list[BatchMessage] | None:
        """Wait for the next batch to decide on, or ``None`` when going silent."""
        timeout = max(0.001, float(round_seconds))
        silent_limit = max(1, int(silent_rounds))
        while True:
            if self.closed:
                return None
            if self.immediate_flush:
                self.immediate_flush = False
                batch = self._take_pending()
                if batch:
                    self.state = STATE_ACTIVE
                    self.empty_rounds = 0
                    self.decision_count += 1
                    return batch
                continue

            wake = self._wake_event()
            wake.clear()
            try:
                await asyncio.wait_for(wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

            if self.closed:
                return None
            if self.immediate_flush:
                continue

            batch = self._take_pending()
            if batch:
                self.state = STATE_ACTIVE
                self.empty_rounds = 0
                self.decision_count += 1
                return batch

            self.empty_rounds += 1
            if self.empty_rounds >= silent_limit:
                self.state = STATE_SILENT
                return None

    # ------------------------------------------------------------------
    # Shutdown / introspection
    # ------------------------------------------------------------------
    def close(self) -> None:
        self.closed = True
        self.pending.clear()
        self.immediate_flush = False
        if self._wake is not None:
            try:
                self._wake.set()
            except RuntimeError:  # pragma: no cover - no running loop
                pass

    def snapshot(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "state": self.state,
            "pending": len(self.pending),
            "empty_rounds": self.empty_rounds,
            "leader_active": self.leader_active,
            "inflight": self.inflight,
            "immediate_flush": self.immediate_flush,
            "decision_count": self.decision_count,
            "dropped_count": self.dropped_count,
        }
