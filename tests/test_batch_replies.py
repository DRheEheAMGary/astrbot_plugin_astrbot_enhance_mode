from __future__ import annotations

import asyncio
import json
import time

import pytest

from astrbot.api.event import MessageEventResult
from astrbot.api.message_components import At, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest

from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import BATCH_PAYLOAD_EXTRA
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    ActiveReplyConfig,
    BatchReplyConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeSender:
    def __init__(self, nickname: str = "Alice") -> None:
        self.nickname = nickname


class _FakeMessageObj:
    def __init__(self, message_id: str, components: list) -> None:
        self.message_id = message_id
        self.sender = _FakeSender()
        self.message = components


class _FakeClient:
    def __init__(self, cached: set[int] | None = None) -> None:
        self.cached = cached or set()

    def get_message(self, message_id: int):
        if message_id in self.cached:
            return object()
        raise KeyError(message_id)


class _FakeEvent:
    def __init__(
        self,
        *,
        origin: str = "discord/GroupMessage/chan-1",
        message_id: str = "100",
        text: str = "hello",
        components: list | None = None,
        wake: bool = False,
        client: _FakeClient | None = None,
    ) -> None:
        self.unified_msg_origin = origin
        self.message_str = text
        self.session_id = "session-1"
        self.is_at_or_wake_command = wake
        self.call_llm = False
        self.message_obj = _FakeMessageObj(
            message_id, components if components is not None else [Plain(text=text)]
        )
        self.client = client or _FakeClient()
        self.extra: dict[str, object] = {}
        self.requests: list[dict] = []
        self.sent: list[list] = []
        self._result: MessageEventResult | None = None

    # --- event api used by the plugin ---
    def get_message_type(self) -> MessageType:
        return MessageType.GROUP_MESSAGE

    def get_sender_id(self) -> str:
        return "42"

    def is_admin(self) -> bool:
        return False

    def get_platform_name(self) -> str:
        return "discord"

    def get_platform_id(self) -> str:
        return "discord-main"

    def get_messages(self) -> list:
        return self.message_obj.message

    def get_extra(self, key: str, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key: str, value) -> None:
        self.extra[key] = value

    def should_call_llm(self, call_llm: bool) -> None:
        self.call_llm = call_llm

    def request_llm(self, **kwargs) -> ProviderRequest:
        self.requests.append(kwargs)
        return ProviderRequest(prompt=kwargs.get("prompt", ""))

    async def send(self, message) -> None:
        self.sent.append(list(message.chain))

    def get_result(self) -> MessageEventResult | None:
        return self._result

    def set_result(self, result: MessageEventResult) -> None:
        self._result = result


def _config(
    *,
    batch_enable: bool = True,
    batch_round: float = 0.01,
    silent_rounds: int = 3,
    history_enable: bool = False,
    active_reply_enable: bool = False,
    reply_interval_ms: int = 0,
    max_pending: int = 10,
) -> PluginConfig:
    return PluginConfig(
        group_history=GroupHistoryEnhancementConfig(enable=history_enable),
        active_reply=ActiveReplyConfig(enable=active_reply_enable),
        batch_reply=BatchReplyConfig(
            enable=batch_enable,
            round_seconds=batch_round,
            silent_rounds=silent_rounds,
            max_pending=max_pending,
            reply_interval_ms=reply_interval_ms,
        ),
        group_features=GroupFeatureEnhancementConfig(react_mode_enable=True),
    )


class _FakeConversation:
    def __init__(self, cid: str = "cid-1") -> None:
        self.cid = cid
        self.history = "[]"
        self.persona_id = ""


class _FakeConversationManager:
    def __init__(self) -> None:
        self.conversation = _FakeConversation()
        self.calls: list[str] = []

    async def get_curr_conversation_id(self, origin: str) -> str:
        self.calls.append("get_curr_conversation_id")
        return self.conversation.cid

    async def new_conversation(self, origin: str, platform_id: str = "") -> str:
        self.calls.append("new_conversation")
        return self.conversation.cid

    async def get_conversation(self, origin: str, cid: str):
        self.calls.append("get_conversation")
        return self.conversation


class _FakeContext:
    def __init__(self, platform_settings: dict | None = None) -> None:
        self.conversation_manager = _FakeConversationManager()
        self._platform_settings = platform_settings or {}

    def get_config(self, umo: str | None = None) -> dict:
        return {"platform_settings": self._platform_settings}


def _build_plugin(
    cfg: PluginConfig, context: _FakeContext | None = None
) -> Main:
    plugin = Main.__new__(Main)
    plugin.runtime = RuntimeState()
    plugin.context = context or _FakeContext()
    plugin._cfg = lambda: cfg

    async def _persona(_event):
        return "tester", "be a helpful tester"

    plugin._resolve_persona_mask = _persona

    recorded: list[str] = []

    async def _record(_event, _cfg):
        recorded.append(_event.message_obj.message_id)

    plugin._record_message = _record
    plugin.recorded_messages = recorded  # type: ignore[attr-defined]
    return plugin


# ---------------------------------------------------------------------------
# request building / round scheduling through the real handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_message_is_batched_into_one_llm_request() -> None:
    plugin = _build_plugin(_config())
    event = _FakeEvent(message_id="100", text="hello there")

    agen = plugin.on_group_message(event)
    request = await anext(agen)

    assert isinstance(request, ProviderRequest)
    assert len(event.requests) == 1
    payload = event.get_extra(BATCH_PAYLOAD_EXTRA)
    assert payload == [
        {
            "message_id": "100",
            "user": "Alice/42",
            "content": "hello there",
            "time": payload[0]["time"],
        }
    ]
    prompt = event.requests[0]["prompt"]
    assert "100" in prompt
    assert "exactly ONE reply" in prompt
    assert '"replies"' not in prompt
    # Conversation is bound, so persona comes from the AstrBot pipeline instead of
    # the manual prompt block (fallback is covered in test_batch_history).
    assert "tester" not in prompt
    assert event.requests[0]["conversation"] is not None
    assert event.get_extra("enable_streaming") is False
    assert event.call_llm is True  # default LLM chain is bypassed for batch decisions
    await agen.aclose()


@pytest.mark.asyncio
async def test_wake_message_flushes_immediately_without_waiting_round() -> None:
    plugin = _build_plugin(_config(batch_round=30.0))
    event = _FakeEvent(message_id="777", text="@bot hi", wake=True)

    started = time.monotonic()
    agen = plugin.on_group_message(event)
    request = await anext(agen)
    elapsed = time.monotonic() - started

    assert request is not None
    assert elapsed < 2.0
    assert event.call_llm is True
    payload = event.get_extra(BATCH_PAYLOAD_EXTRA)
    assert [item["message_id"] for item in payload] == ["777"]
    await agen.aclose()


@pytest.mark.asyncio
async def test_at_bot_only_message_still_triggers_flush() -> None:
    plugin = _build_plugin(_config(batch_round=30.0))
    event = _FakeEvent(
        message_id="888",
        text="",
        components=[At(qq="bot-1", name="Bot")],
        wake=True,
    )

    agen = plugin.on_group_message(event)
    await anext(agen)
    payload = event.get_extra(BATCH_PAYLOAD_EXTRA)
    assert payload[0]["content"] == "[At: Bot]"
    await agen.aclose()


@pytest.mark.asyncio
async def test_wake_message_during_inflight_flushes_after_call_finishes() -> None:
    plugin = _build_plugin(_config(batch_round=0.01))
    leader = _FakeEvent(message_id="100", text="first")
    agen = plugin.on_group_message(leader)
    await anext(agen)

    coordinator = plugin.runtime.batch_coordinators[leader.unified_msg_origin]
    assert coordinator.inflight is True

    # @ arrives while the model call is in flight: queued, not a concurrent call.
    wake_event = _FakeEvent(message_id="200", text="@bot ping", wake=True)
    async for _ in plugin.on_group_message(wake_event):
        raise AssertionError("wake event must not produce its own LLM request")
    assert wake_event.requests == []
    assert wake_event.call_llm is True
    assert len(coordinator.pending) == 1

    started = time.monotonic()
    request = await anext(agen)
    elapsed = time.monotonic() - started
    assert request is not None
    assert elapsed < 2.0
    payload = leader.get_extra(BATCH_PAYLOAD_EXTRA)
    assert [item["message_id"] for item in payload] == ["200"]
    await agen.aclose()


@pytest.mark.asyncio
async def test_silent_rounds_release_leader_without_extra_requests() -> None:
    plugin = _build_plugin(_config(batch_round=0.01, silent_rounds=2))
    event = _FakeEvent(message_id="100", text="hi")

    requests: list[ProviderRequest] = []
    started = time.monotonic()
    async for request in plugin.on_group_message(event):
        requests.append(request)
    elapsed = time.monotonic() - started

    assert len(requests) == 1
    assert len(event.requests) == 1
    # 2 empty rounds * 0.01s after the single decision.
    assert elapsed >= 0.02
    coordinator = plugin.runtime.batch_coordinators[event.unified_msg_origin]
    assert coordinator.leader_active is False
    assert coordinator.state == "SILENT"


@pytest.mark.asyncio
async def test_batch_disabled_keeps_legacy_path_and_creates_no_coordinator() -> None:
    plugin = _build_plugin(_config(batch_enable=False, history_enable=True))

    async def _no_active_reply(_event, _cfg):
        return False

    plugin._need_active_reply = _no_active_reply
    event = _FakeEvent(message_id="100", text="hi")

    requests = [request async for request in plugin.on_group_message(event)]
    assert requests == []
    assert event.requests == []
    assert plugin.runtime.batch_coordinators == {}
    assert plugin.recorded_messages == ["100"]


# ---------------------------------------------------------------------------
# decorating result: multi reply, quote, splitting, silence
# ---------------------------------------------------------------------------


def _decorating_event(
    raw_text: str,
    payload: list[dict[str, str]],
    *,
    cached: set[int] | None = None,
) -> _FakeEvent:
    event = _FakeEvent(
        message_id="100",
        text="hi",
        client=_FakeClient(cached if cached is not None else {100, 101}),
    )
    event.set_extra(BATCH_PAYLOAD_EXTRA, payload)
    event.set_result(MessageEventResult().message(raw_text))
    return event


@pytest.mark.asyncio
async def test_decorating_sends_one_plain_reply_per_batch() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [
        {"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"},
        {"message_id": "101", "user": "B/2", "content": "b", "time": "12:01"},
    ]
    # Legacy replies/message_id/quote fields on purpose: only the first text is used.
    raw = (
        '{"replies":['
        '{"message_id":"100","text":"第一行\\n第二行","quote":true},'
        '{"message_id":"101","text":"second legacy reply","quote":false}'
        "]}"
    )
    event = _decorating_event(raw, payload)

    await plugin.parse_tags(event)

    assert event.get_result().chain == []
    assert len(event.sent) == 1
    chain = event.sent[0]
    assert not any(isinstance(c, Reply) for c in chain)
    assert any(isinstance(c, Plain) and c.text == "第一行\n第二行" for c in chain)
    assert all(
        not (isinstance(c, Plain) and "second legacy reply" in c.text) for c in chain
    )


@pytest.mark.asyncio
async def test_decorating_sends_plain_text_model_output_directly() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [{"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"}]
    event = _decorating_event("直接回复\n第二行", payload)

    await plugin.parse_tags(event)

    assert event.get_result().chain == []
    assert len(event.sent) == 1
    chain = event.sent[0]
    assert not any(isinstance(c, Reply) for c in chain)
    assert any(isinstance(c, Plain) and c.text == "直接回复\n第二行" for c in chain)


@pytest.mark.asyncio
async def test_decorating_silent_decision_sends_nothing() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [{"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"}]
    event = _decorating_event('{"replies":[]}', payload)

    await plugin.parse_tags(event)

    assert event.get_result().chain == []
    assert event.sent == []


@pytest.mark.asyncio
async def test_decorating_long_reply_is_split_into_discord_safe_chunks() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [{"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"}]
    long_text = "intro line\n" + ("x" * 2500) + "\ntail line"
    raw = json.dumps(
        {"replies": [{"message_id": "100", "text": long_text, "quote": True}]},
        ensure_ascii=False,
    )
    event = _decorating_event(raw, payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 2
    texts = [
        "".join(c.text for c in chain if isinstance(c, Plain)) for chain in event.sent
    ]
    assert all(len(text) <= 2000 for text in texts)
    assert "".join(texts) == long_text
    for chain in event.sent:
        assert not any(isinstance(c, Reply) for c in chain)


@pytest.mark.asyncio
async def test_decorating_ignores_legacy_quote_even_when_reference_cached() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [{"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"}]
    raw = '{"replies":[{"message_id":"100","text":"hello","quote":true}]}'
    # The reference IS resolvable, but batch replies must stay plain (no Reply).
    event = _decorating_event(raw, payload, cached={100})

    await plugin.parse_tags(event)

    assert len(event.sent) == 1
    assert not any(isinstance(c, Reply) for c in event.sent[0])
    assert any(isinstance(c, Plain) and c.text == "hello" for c in event.sent[0])


@pytest.mark.asyncio
async def test_decorating_parses_mention_tags_inside_batch_text() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [{"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"}]
    raw = (
        '{"replies":[{"message_id":"100","text":'
        '"<mention id=\\"42\\"/> hello","quote":true}]}'
    )
    event = _decorating_event(raw, payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 1
    chain = event.sent[0]
    assert not any(isinstance(c, Reply) for c in chain)
    assert any(isinstance(c, At) and str(c.qq) == "42" for c in chain)


@pytest.mark.asyncio
async def test_legacy_refuse_tag_still_clears_result_without_batch_payload() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    result = MessageEventResult().message("<refuse/>")
    event.set_result(result)
    await plugin.parse_tags(event)
    assert event.get_result().chain == []
    assert event.get_extra("_enhance_refused_reply") is True
    assert event.sent == []


# ---------------------------------------------------------------------------
# history write-back / llm request hook
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, text: str) -> None:
        self.completion_text = text


@pytest.mark.asyncio
async def test_batch_response_history_records_reply_text_not_raw_json() -> None:
    cfg = _config(history_enable=True)
    plugin = _build_plugin(cfg)
    event = _FakeEvent(message_id="100", text="hi")
    plugin.runtime.session_chats[event.unified_msg_origin] = ["[Alice/42/12:00:00]: hi"]
    payload = [{"message_id": "100", "user": "A/1", "content": "hi", "time": "12:00"}]
    event.set_extra(BATCH_PAYLOAD_EXTRA, payload)
    raw = (
        '{"replies":['
        '{"message_id":"100","text":"回复一\\n续行","quote":true},'
        '{"message_id":"100","text":"回复二","quote":false}'
        "]}"
    )

    await plugin.record_bot_response(event, _FakeResp(raw))

    chats = plugin.runtime.session_chats[event.unified_msg_origin]
    assert len(chats) == 2  # one user line + one assistant reply line
    assert chats[1].endswith("回复一\n续行") and chats[1].startswith("[You/")
    assert all("{" not in line and "replies" not in line for line in chats)
    assert all("回复二" not in line for line in chats)


@pytest.mark.asyncio
async def test_batch_silent_response_writes_no_history() -> None:
    cfg = _config(history_enable=True)
    plugin = _build_plugin(cfg)
    event = _FakeEvent(message_id="100", text="hi")
    plugin.runtime.session_chats[event.unified_msg_origin] = ["existing"]
    event.set_extra(
        BATCH_PAYLOAD_EXTRA,
        [{"message_id": "100", "user": "A/1", "content": "hi", "time": "12:00"}],
    )

    await plugin.record_bot_response(event, _FakeResp('{"replies":[]}'))

    assert plugin.runtime.session_chats[event.unified_msg_origin] == ["existing"]


@pytest.mark.asyncio
async def test_batch_llm_request_hook_keeps_contexts_and_prompt() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    event.set_extra(BATCH_PAYLOAD_EXTRA, [{"message_id": "100"}])
    contexts = [{"role": "user", "content": "previous turn"}]
    request = ProviderRequest(prompt="batch prompt", contexts=contexts)

    await plugin.inject_group_context(event, request)

    assert request.prompt == "batch prompt"
    assert request.contexts == contexts


@pytest.mark.asyncio
async def test_terminate_releases_held_coordinators() -> None:
    plugin = _build_plugin(_config(batch_round=0.01))
    plugin._stop_memory_rag_webui = _noop  # type: ignore[attr-defined]
    event = _FakeEvent(message_id="100", text="hi")
    agen = plugin.on_group_message(event)
    await anext(agen)
    coordinator = plugin.runtime.batch_coordinators[event.unified_msg_origin]

    await plugin.terminate()

    assert coordinator.closed is True
    assert plugin.runtime.batch_coordinators == {}
    await agen.aclose()


async def _noop(*_args, **_kwargs) -> None:
    return None


@pytest.mark.asyncio
async def test_high_frequency_messages_never_start_parallel_model_calls() -> None:
    plugin = _build_plugin(_config(batch_round=0.01))
    leader = _FakeEvent(message_id="100", text="leader")
    agen = plugin.on_group_message(leader)
    await anext(agen)

    # 20 messages arrive while the leader is deciding.
    for index in range(20):
        follower = _FakeEvent(message_id=str(200 + index), text=f"m{index}")
        assert [item async for item in plugin.on_group_message(follower)] == []
        assert follower.requests == []

    coordinator = plugin.runtime.batch_coordinators[leader.unified_msg_origin]
    assert len(coordinator.pending) == 10  # max_pending=10, oldest dropped
    assert coordinator.dropped_count == 10
    await agen.aclose()


@pytest.mark.asyncio
async def test_command_event_does_not_enter_batch_decision() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="/enhance rag-webui")

    class CommandFilter:
        pass

    handler = type("Handler", (), {"event_filters": [CommandFilter()]})()
    event.set_extra("activated_handlers", [handler])

    requests = [request async for request in plugin.on_group_message(event)]

    assert requests == []
    assert plugin.runtime.batch_coordinators == {}
    assert plugin.recorded_messages == ["100"]


@pytest.mark.asyncio
async def test_slash_command_interaction_does_not_enter_batch_decision() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="")
    event.message_obj.message = []
    event.is_at_or_wake_command = True
    event.is_slash_command = lambda: True  # type: ignore[attr-defined]

    requests = [request async for request in plugin.on_group_message(event)]

    assert requests == []
    assert plugin.runtime.batch_coordinators == {}


@pytest.mark.asyncio
async def test_decorating_parses_mention_in_plain_text_output() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    payload = [{"message_id": "100", "user": "A/1", "content": "a", "time": "12:00"}]
    # New contract: plain text output that mentions a user from the batch.
    event = _decorating_event('<mention id="42"/> 你好呀', payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 1
    chain = event.sent[0]
    assert not any(isinstance(c, Reply) for c in chain)
    assert any(isinstance(c, At) and str(c.qq) == "42" for c in chain)
    assert any(isinstance(c, Plain) and "你好呀" in c.text for c in chain)
