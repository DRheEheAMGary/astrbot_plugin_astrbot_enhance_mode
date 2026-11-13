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
import math
import random
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

STATE_ACTIVE = "ACTIVE"
STATE_SILENT = "SILENT"

DISCORD_MAX_MESSAGE_LENGTH = 2000
DEFAULT_MAX_REPLIES = 5
# Upper bound of images attached to one batch decision (per round).
BATCH_MAX_IMAGES = 8
# Upper bound of on-demand history images a model may fetch in one decision.
BATCH_HISTORY_IMAGE_LIMIT = 3
# Event extra tracking images already fetched via enhance_use_image in a batch run.
BATCH_FETCHED_IMAGES_EXTRA = "_enhance_batch_fetched_images"

# Event extra used to tell the LLM-request path which batch is being decided.
BATCH_PAYLOAD_EXTRA = "_enhance_batch_payload"
# Event extra holding the conversation bound to a batch request (for history write-back).
BATCH_CONVERSATION_EXTRA = "_enhance_batch_conversation"
# Marker embedded in the batch prompt so the history rewrite can find the user turn.
BATCH_PROMPT_MARKER = "New messages collected in the current round"

SEGMENT_EXCLUDED_PLATFORMS = {
    "qq_official_webhook",
    "weixin_official_account",
    "dingtalk",
}
DEFAULT_SEGMENT_LOG_BASE = 2.6
DEFAULT_SEGMENT_INTERVAL = (1.5, 3.5)
DEFAULT_SEGMENT_REGEX = r".*?[。？！~…]+|.+$"
DEFAULT_SPLIT_WORDS = ("。", "？", "！", "~", "…")

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
    """One collected user message that participates in a batch decision.

    ``images`` keeps the raw image references (URL/local path/base64) collected
    from the message in appearance order. They are internal only and never
    exposed through :meth:`to_payload` (the model receives them as ordered
    ``image_urls`` instead).
    """

    message_id: str
    user: str
    content: str
    time: str
    wake: bool = False
    images: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "user": self.user,
            "content": self.content,
            "time": self.time,
        }


@dataclass(frozen=True)
class ReplySpec:
    """One model-produced reply parsed from the batch JSON contract.

    Batch replies are sent as plain messages: they carry no quote reference and
    no target message id. Legacy outputs that still include ``message_id`` or
    ``quote`` fields are accepted and the extra fields are ignored.
    """

    text: str


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
    max_replies: int = DEFAULT_MAX_REPLIES,
) -> list[ReplySpec]:
    """Parse the batch decision JSON into plain reply texts.

    Only ``text`` is used. Legacy ``message_id``/``quote`` fields are ignored so
    older model outputs keep working without producing a quote reference.
    Invalid/empty entries are skipped.
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
        replies.append(ReplySpec(text=text))
    return replies


def parse_batch_reply_text(raw_text: str | None) -> str:
    """Normalize model output into the single plain reply text for one batch.

    The current contract is plain text (one reply for the whole batch). Legacy
    outputs that still use ``{"replies": [...]}`` are accepted: only the first
    entry is used and ``message_id``/``quote`` fields are ignored. Returns an
    empty string when the batch should stay silent.
    """
    if not raw_text:
        return ""
    text = str(raw_text).strip()
    if not text:
        return ""
    if text.startswith("<refuse/>"):
        return ""

    payload = extract_json_object(text)
    if isinstance(payload, dict):
        raw_replies = payload.get("replies")
        if isinstance(raw_replies, list):
            replies = parse_batch_replies(text, max_replies=1)
            return replies[0].text if replies else ""
        single_text = payload.get("text")
        if isinstance(single_text, str):
            return single_text.strip()
        return ""
    return text


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
    mention_parse: bool,
    persona_name: str = "",
    persona_mask: str = "",
    max_history_chars: int = 0,
    image_count: int = 0,
    history_images_available: bool = False,
    history_image_limit: int = BATCH_HISTORY_IMAGE_LIMIT,
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
    image_rule = ""
    if image_count > 0:
        image_rule = (
            f"- This request attaches {image_count} image(s) in message order. "
            '"[Image #k]" in the messages below refers to the k-th attached image; '
            "reference images only by the numbers that appear and never invent numbers.\n"
        )
    history_image_rule = ""
    if history_images_available:
        history_image_rule = (
            "- Some historical messages above contain `[Image]` markers (possibly with "
            "`[images: N]` at the end of the line) and have a header containing "
            '`#msg<message_id>`. If you need to look at a historical image to answer '
            "well, call the tool "
            '`enhance_use_image(message_id="<message_id>", image_index=<k>, '
            "attach_to_model=true, write_to_history=false)` where `k` starts at 1. "
            f"You may fetch at most {max(0, int(history_image_limit))} images in this round; "
            "do not invent or guess image contents without fetching, and do not fetch "
            "images for the current batch (they are already attached as `[Image #k]`).\n"
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
        f"{BATCH_PROMPT_MARKER} (JSON array):\n"
        f"{batch_json}\n\n"
        "Decide, in character, whether to respond to this batch of new messages. "
        "Write exactly ONE reply for the whole batch, as plain message text. "
        "Reply directly; do NOT output JSON, a replies array, message ids, quote "
        "markers, or any wrapper, and do NOT quote the original messages.\n"
        "Rules:\n"
        "- Output the reply text directly; it may contain \\n line breaks "
        "(preserved when sent).\n"
        "- Keep the reply under 2000 characters.\n"
        f"{image_rule}"
        f"{history_image_rule}"
        f"{mention_rule}"
        "- Use the SAME language as the chatroom.\n"
        "- If you decide not to reply, output exactly <refuse/> and nothing else.\n"
        "- Only output your response and no other information."
    )


# ---------------------------------------------------------------------------
# AstrBot ``platform_settings.segmented_reply`` replication
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _parse_interval(raw: Any) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        values = [_coerce_float(raw[0], DEFAULT_SEGMENT_INTERVAL[0]), _coerce_float(raw[1], DEFAULT_SEGMENT_INTERVAL[1])]
        return float(values[0]), float(values[1])
    if isinstance(raw, str):
        parts = raw.replace(" ", "").split(",")
        if len(parts) >= 2:
            return _coerce_float(parts[0], 1.5), _coerce_float(parts[1], 3.5)
    return DEFAULT_SEGMENT_INTERVAL


def _parse_split_words(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, str):
        items = [item.strip() for item in re.split(r"[,，\n]", raw)]
    elif isinstance(raw, (list, tuple, set)):
        items = [str(item).strip() for item in raw]
    else:
        items = []
    words = tuple(item for item in items if item)
    return words or DEFAULT_SPLIT_WORDS


@dataclass(frozen=True)
class SegmentedReplyOptions:
    """Snapshot of AstrBot ``platform_settings.segmented_reply``."""

    enable: bool = False
    only_llm_result: bool = False
    words_count_threshold: int = 1000
    split_mode: str = "regex"
    regex: str = DEFAULT_SEGMENT_REGEX
    split_words: tuple[str, ...] = DEFAULT_SPLIT_WORDS
    content_cleanup_rule: str = ""
    interval_method: str = "log"
    interval_min: float = 1.5
    interval_max: float = 3.5
    log_base: float = DEFAULT_SEGMENT_LOG_BASE


def parse_segmented_reply_options(raw: Any) -> SegmentedReplyOptions:
    """Parse/normalize AstrBot segmented_reply settings (see ResultDecorateStage)."""
    if not isinstance(raw, dict):
        return SegmentedReplyOptions()

    interval_min, interval_max = _parse_interval(raw.get("interval"))
    split_mode = str(raw.get("split_mode") or "regex").strip().lower()
    if split_mode not in {"words", "regex"}:
        split_mode = "regex"
    interval_method = str(raw.get("interval_method") or "log").strip().lower()
    if interval_method not in {"log", "random"}:
        interval_method = "log"
    log_base = _coerce_float(raw.get("log_base"), DEFAULT_SEGMENT_LOG_BASE)
    if log_base <= 1:
        log_base = DEFAULT_SEGMENT_LOG_BASE
    regex_value = str(raw.get("regex") or "").strip() or DEFAULT_SEGMENT_REGEX
    return SegmentedReplyOptions(
        enable=_coerce_bool(raw.get("enable"), False),
        only_llm_result=_coerce_bool(raw.get("only_llm_result"), False),
        words_count_threshold=max(0, _coerce_int(raw.get("words_count_threshold"), 1000)),
        split_mode=split_mode,
        regex=regex_value,
        split_words=_parse_split_words(raw.get("split_words")),
        content_cleanup_rule=str(raw.get("content_cleanup_rule") or ""),
        interval_method=interval_method,
        interval_min=interval_min,
        interval_max=interval_max,
        log_base=log_base,
    )


def segment_word_count(text: str) -> int:
    """Word counting rule used by AstrBot's segmented reply interval."""
    if all(ord(char) < 128 for char in text):
        return len(text.split())
    return len([char for char in text if char.isalnum()])


def _split_text_by_words(text: str, split_words: tuple[str, ...]) -> list[str]:
    """Replicate ResultDecorateStage._split_text_by_words."""
    escaped = sorted([re.escape(word) for word in split_words], key=len, reverse=True)
    if not escaped:
        return [text]
    pattern = re.compile(f"(.*?({'|'.join(escaped)})|.+$)", re.DOTALL)
    segments = pattern.findall(text)
    result: list[str] = []
    for segment in segments:
        if isinstance(segment, tuple):
            content = segment[0]
            if not isinstance(content, str):
                continue
            for word in split_words:
                if content.endswith(word):
                    content = content[: -len(word)]
                    break
            if content.strip():
                result.append(content)
        elif segment and segment.strip():
            result.append(segment)
    return result if result else [text]


def split_segment_text(text: str, options: SegmentedReplyOptions) -> list[str]:
    """Split one Plain text the same way AstrBot's ResultDecorateStage does."""
    if not text or not text.strip():
        return []
    if len(text) > options.words_count_threshold:
        # Too long to segment: AstrBot keeps the whole component.
        return [text]
    if options.split_mode == "words":
        segments = _split_text_by_words(text, options.split_words)
    else:
        try:
            segments = re.findall(options.regex, text, re.DOTALL | re.MULTILINE)
        except re.error:
            segments = re.findall(
                DEFAULT_SEGMENT_REGEX, text, re.DOTALL | re.MULTILINE
            )
    if not segments:
        return [text]
    cleaned: list[str] = []
    for segment in segments:
        if not isinstance(segment, str):
            continue
        if options.content_cleanup_rule:
            try:
                segment = re.sub(options.content_cleanup_rule, "", segment)
            except re.error:
                pass
        segment = segment.strip()
        if segment:
            cleaned.append(segment)
    return cleaned


def calc_segment_interval(
    text: str | None,
    options: SegmentedReplyOptions,
    rng: random.Random | None = None,
) -> float:
    """Replicate RespondStage._calc_comp_interval (``text=None`` for non-Plain)."""
    generator = rng or random
    if options.interval_method == "log":
        if text is None:
            return generator.uniform(1, 1.75)
        word_count = segment_word_count(text)
        base = math.log(word_count + 1, options.log_base)
        return generator.uniform(base, base + 0.5)
    return generator.uniform(options.interval_min, options.interval_max)


# ---------------------------------------------------------------------------
# Readable history formatting
# ---------------------------------------------------------------------------


def _clip_text(text: str, limit: int = 500) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_readable_batch_user_text(batch_payload: Iterable[dict[str, str]]) -> str:
    """Human readable user turn written into AstrBot conversation history."""
    lines = ["（本轮批量消息）"]
    for item in batch_payload:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user") or "unknown")
        time_text = str(item.get("time") or "")
        content = _clip_text(str(item.get("content") or "").strip())
        prefix = f"[{user} {time_text}]".replace(" ]", "]")
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def build_readable_batch_assistant_text(reply_text: str | None) -> str:
    """One-line assistant turn written into AstrBot conversation history."""
    text = (reply_text or "").strip()
    if not text:
        return "（本轮未回复）"
    return _clip_text(text, 8000)


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
