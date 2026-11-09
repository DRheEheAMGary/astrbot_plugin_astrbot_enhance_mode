from __future__ import annotations

import pytest

from astrbot.core.agent.message import Message

from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import (
    BATCH_CONVERSATION_EXTRA,
    BATCH_PAYLOAD_EXTRA,
    BATCH_PROMPT_MARKER,
    build_readable_batch_assistant_text,
    build_readable_batch_user_text,
)
from astrbot_plugin_astrbot_enhance_mode.main import Main
from astrbot_plugin_astrbot_enhance_mode.plugin_config import (
    ActiveReplyConfig,
    BatchReplyConfig,
    GroupFeatureEnhancementConfig,
    GroupHistoryEnhancementConfig,
    PluginConfig,
)
from astrbot_plugin_astrbot_enhance_mode.runtime_state import RuntimeState

from test_batch_replies import (
    _FakeContext,
    _FakeConversationManager,
    _FakeEvent,
    _FakeResp,
    _build_plugin,
    _config,
)


# ---------------------------------------------------------------------------
# conversation binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_request_binds_current_conversation() -> None:
    plugin = _build_plugin(_config())
    event = _FakeEvent(message_id="100", text="hello")

    agen = plugin.on_group_message(event)
    await anext(agen)
    try:
        assert len(event.requests) == 1
        conversation = event.requests[0]["conversation"]
        assert conversation is plugin.context.conversation_manager.conversation
        assert event.get_extra(BATCH_CONVERSATION_EXTRA) is conversation
        assert "get_curr_conversation_id" in plugin.context.conversation_manager.calls
        assert "get_conversation" in plugin.context.conversation_manager.calls
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_batch_request_creates_conversation_when_missing() -> None:
    class _NoCurrentManager(_FakeConversationManager):
        async def get_curr_conversation_id(self, origin: str) -> str:
            return ""

    class _NoCurrentContext(_FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.conversation_manager = _NoCurrentManager()

    plugin = _build_plugin(_config(), context=_NoCurrentContext())
    event = _FakeEvent(message_id="100", text="hello")

    agen = plugin.on_group_message(event)
    await anext(agen)
    try:
        assert event.requests[0]["conversation"] is not None
        assert "new_conversation" in plugin.context.conversation_manager.calls
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_batch_request_falls_back_when_conversation_unavailable() -> None:
    class _BrokenManager(_FakeConversationManager):
        async def get_curr_conversation_id(self, origin: str) -> str:
            raise RuntimeError("db down")

    class _BrokenContext(_FakeContext):
        def __init__(self) -> None:
            super().__init__()
            self.conversation_manager = _BrokenManager()

    plugin = _build_plugin(_config(), context=_BrokenContext())
    event = _FakeEvent(message_id="100", text="hello")

    agen = plugin.on_group_message(event)
    await anext(agen)
    try:
        assert event.requests[0]["conversation"] is None
        prompt = event.requests[0]["prompt"]
        # Manual persona fallback is injected when no conversation is bound.
        assert "tester" in prompt
    finally:
        await agen.aclose()


# ---------------------------------------------------------------------------
# readable history write-back
# ---------------------------------------------------------------------------


def _build_run_context(prompt: str, raw_response: str):
    class _RunContext:
        def __init__(self) -> None:
            self.messages = [
                Message(role="system", content="system prompt"),
                Message(
                    role="user",
                    content=(
                        "You are now in a chatroom.\n\n"
                        f"{BATCH_PROMPT_MARKER} (JSON array):\n"
                        '[{"message_id":"100"}]\n\nOutput ONLY JSON.'
                    ),
                ),
                Message(role="assistant", content=raw_response),
            ]

    return _RunContext()


def _batch_payload() -> list[dict[str, str]]:
    return [
        {
            "message_id": "100",
            "user": "Alice/42",
            "content": "你好呀",
            "time": "12:00:05",
        },
        {
            "message_id": "101",
            "user": "Bob/43",
            "content": "在吗",
            "time": "12:00:07",
        },
    ]


@pytest.mark.asyncio
async def test_history_rewrite_replaces_prompt_and_json_with_readable_text() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    payload = _batch_payload()
    event.set_extra(BATCH_PAYLOAD_EXTRA, payload)
    event.set_extra(BATCH_CONVERSATION_EXTRA, object())
    raw = (
        '{"replies":['
        '{"message_id":"100","text":"哈哈 你好","quote":true},'
        '{"message_id":"101","text":"在的\\n怎么了","quote":false}'
        "]}"
    )
    run_context = _build_run_context("", raw)

    await plugin.rewrite_batch_history(event, run_context, _FakeResp(raw))

    user_content = run_context.messages[1].content
    assistant_content = run_context.messages[2].content
    assert user_content.startswith("（本轮批量消息）")
    assert "[Alice/42 12:00:05]: 你好呀" in user_content
    assert "[Bob/43 12:00:07]: 在吗" in user_content
    assert assistant_content == "（引用 #msg100）哈哈 你好\n\n（引用 #msg101）在的\n怎么了"
    assert "replies" not in assistant_content
    assert BATCH_PROMPT_MARKER not in user_content


@pytest.mark.asyncio
async def test_history_rewrite_silent_round_is_marked_readable() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    event.set_extra(BATCH_PAYLOAD_EXTRA, _batch_payload())
    event.set_extra(BATCH_CONVERSATION_EXTRA, object())
    run_context = _build_run_context("", '{"replies":[]}')

    await plugin.rewrite_batch_history(event, run_context, _FakeResp('{"replies":[]}'))

    assert run_context.messages[2].content == "（本轮未回复）"


@pytest.mark.asyncio
async def test_history_rewrite_skips_without_batch_payload() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    run_context = _build_run_context("", "plain reply")

    await plugin.rewrite_batch_history(event, run_context, _FakeResp("plain reply"))

    assert BATCH_PROMPT_MARKER in run_context.messages[1].content
    assert run_context.messages[2].content == "plain reply"


@pytest.mark.asyncio
async def test_history_rewrite_skips_without_conversation_binding() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    event.set_extra(BATCH_PAYLOAD_EXTRA, _batch_payload())
    run_context = _build_run_context("", '{"replies":[]}')

    await plugin.rewrite_batch_history(event, run_context, _FakeResp('{"replies":[]}'))

    assert "replies" in run_context.messages[2].content


def test_readable_user_text_is_single_line_per_message() -> None:
    text = build_readable_batch_user_text(_batch_payload())
    lines = text.splitlines()
    assert lines[0] == "（本轮批量消息）"
    assert lines[1].startswith("[Alice/42 12:00:05]:")


def test_readable_assistant_text_falls_back_to_raw_plain_reply() -> None:
    from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import parse_batch_replies

    replies = parse_batch_replies('{"replies":[]}', ["100"])
    assert build_readable_batch_assistant_text(replies, "plain model text") == (
        "plain model text"
    )
    assert build_readable_batch_assistant_text(replies, '{"replies":[]}') == (
        "（本轮未回复）"
    )
