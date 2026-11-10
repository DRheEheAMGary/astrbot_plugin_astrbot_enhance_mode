from __future__ import annotations

import asyncio
import time

import pytest

from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import (
    STATE_ACTIVE,
    STATE_SILENT,
    BatchCoordinator,
    BatchMessage,
    build_batch_prompt,
    extract_json_object,
    normalize_message_id,
    parse_batch_replies,
    parse_batch_reply_text,
    split_discord_text,
)


def _msg(message_id: str, content: str = "hi", wake: bool = False) -> BatchMessage:
    return BatchMessage(
        message_id=message_id,
        user="Alice/42",
        content=content,
        time="12:00:00",
        wake=wake,
    )


# ---------------------------------------------------------------------------
# coordinator state machine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_message_becomes_leader_and_flushes_next_round() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.state == STATE_SILENT
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    assert coordinator.state == STATE_ACTIVE
    assert coordinator.leader_active is True

    batch = await coordinator.next_batch(round_seconds=0.02, silent_rounds=3)
    assert batch is not None
    assert [item.message_id for item in batch] == ["1"]
    assert coordinator.decision_count == 1
    assert len(coordinator.pending) == 0


@pytest.mark.asyncio
async def test_round_collects_all_pending_in_one_decision() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    # Messages that arrive inside the same round are not leaders.
    assert coordinator.submit(_msg("2"), wake=False, max_pending=10) is False
    assert coordinator.submit(_msg("3"), wake=False, max_pending=10) is False

    batch = await coordinator.next_batch(round_seconds=0.02, silent_rounds=3)
    assert batch is not None
    assert [item.message_id for item in batch] == ["1", "2", "3"]
    assert len(coordinator.pending) == 0


@pytest.mark.asyncio
async def test_consecutive_empty_rounds_go_silent_and_stop_decisions() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    batch = await coordinator.next_batch(round_seconds=0.01, silent_rounds=2)
    assert batch is not None

    started = time.monotonic()
    assert await coordinator.next_batch(round_seconds=0.01, silent_rounds=2) is None
    elapsed = time.monotonic() - started
    assert elapsed >= 0.02  # two empty rounds were awaited
    assert coordinator.state == STATE_SILENT
    assert coordinator.empty_rounds == 2
    # No model decision happened for the empty rounds.
    assert coordinator.decision_count == 1
    assert time.monotonic() - started < 2.0


@pytest.mark.asyncio
async def test_new_message_after_silent_wakes_coordinator() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    assert await coordinator.next_batch(round_seconds=0.01, silent_rounds=1) is not None
    assert await coordinator.next_batch(round_seconds=0.01, silent_rounds=1) is None
    assert coordinator.state == STATE_SILENT
    coordinator.end_leader()

    assert coordinator.submit(_msg("2"), wake=False, max_pending=10) is True
    assert coordinator.state == STATE_ACTIVE
    assert coordinator.empty_rounds == 0
    batch = await coordinator.next_batch(round_seconds=0.01, silent_rounds=1)
    assert batch is not None
    assert [item.message_id for item in batch] == ["2"]


@pytest.mark.asyncio
async def test_wake_message_cancels_remaining_round_wait() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    # @ arrives during the round: it must not wait for round_seconds.
    assert coordinator.submit(_msg("2", wake=True), wake=True, max_pending=10) is False

    started = time.monotonic()
    batch = await coordinator.next_batch(round_seconds=30.0, silent_rounds=3)
    elapsed = time.monotonic() - started
    assert elapsed < 2.0
    assert batch is not None
    assert [item.message_id for item in batch] == ["1", "2"]


@pytest.mark.asyncio
async def test_wake_during_inflight_flushes_right_after_call_finishes() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    first = await coordinator.next_batch(round_seconds=0.01, silent_rounds=3)
    assert first is not None
    coordinator.begin_inflight()

    # New messages during the in-flight call are only queued.
    assert coordinator.submit(_msg("2"), wake=False, max_pending=10) is False
    assert coordinator.submit(_msg("3", wake=True), wake=True, max_pending=10) is False
    assert coordinator.inflight is True
    assert len(coordinator.pending) == 2

    coordinator.end_inflight()
    started = time.monotonic()
    second = await coordinator.next_batch(round_seconds=30.0, silent_rounds=3)
    assert time.monotonic() - started < 2.0
    assert second is not None
    assert [item.message_id for item in second] == ["2", "3"]


@pytest.mark.asyncio
async def test_pending_has_upper_bound_and_drops_oldest() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=2) is True
    coordinator.submit(_msg("2"), wake=False, max_pending=2)
    coordinator.submit(_msg("3"), wake=False, max_pending=2)
    coordinator.submit(_msg("4"), wake=False, max_pending=2)
    assert [item.message_id for item in coordinator.pending] == ["3", "4"]
    assert coordinator.dropped_count == 2


@pytest.mark.asyncio
async def test_only_one_leader_and_one_inflight_per_coordinator() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True
    assert coordinator.submit(_msg("2"), wake=False, max_pending=10) is False
    assert coordinator.submit(_msg("3"), wake=False, max_pending=10) is False
    await coordinator.next_batch(round_seconds=0.01, silent_rounds=3)
    coordinator.begin_inflight()
    assert coordinator.inflight is True
    assert coordinator.submit(_msg("4"), wake=False, max_pending=10) is False
    coordinator.end_inflight()
    assert coordinator.inflight is False


@pytest.mark.asyncio
async def test_close_releases_waiter_without_decision() -> None:
    coordinator = BatchCoordinator("origin")
    assert coordinator.submit(_msg("1"), wake=False, max_pending=10) is True

    async def _close_soon() -> None:
        await asyncio.sleep(0.01)
        coordinator.close()

    closer = asyncio.create_task(_close_soon())
    started = time.monotonic()
    batch = await coordinator.next_batch(round_seconds=30.0, silent_rounds=3)
    await closer
    assert batch is None
    assert coordinator.closed is True
    assert time.monotonic() - started < 2.0


# ---------------------------------------------------------------------------
# JSON parsing / splitting
# ---------------------------------------------------------------------------


def test_normalize_message_id() -> None:
    assert normalize_message_id("#msg123") == "123"
    assert normalize_message_id("msg456") == "456"
    assert normalize_message_id(" 789 ") == "789"
    assert normalize_message_id(None) == ""


def test_parse_batch_replies_valid_multi_reply_preserves_newlines() -> None:
    raw = (
        '{"replies":['
        '{"text":"第一行\\n第二行"},'
        '{"text":"第二条"}'
        "]}"
    )
    replies = parse_batch_replies(raw, max_replies=5)
    assert [reply.text for reply in replies] == ["第一行\n第二行", "第二条"]


def test_parse_batch_replies_ignores_legacy_quote_fields() -> None:
    raw = (
        '{"replies":['
        '{"message_id":"100","text":"旧格式一条","quote":true},'
        '{"message_id":"101","text":"旧格式两条","quote":false}'
        "]}"
    )
    replies = parse_batch_replies(raw, max_replies=5)
    assert [reply.text for reply in replies] == ["旧格式一条", "旧格式两条"]
    for reply in replies:
        assert not hasattr(reply, "quote")
        assert not hasattr(reply, "message_id")


def test_parse_batch_replies_handles_markdown_fence_and_literal_newline() -> None:
    raw = '```json\n{"replies":[{"text":"a\nb"}]}\n```'
    replies = parse_batch_replies(raw, max_replies=5)
    assert len(replies) == 1
    assert replies[0].text == "a\nb"


def test_parse_batch_replies_silent_and_garbage() -> None:
    assert parse_batch_replies('{"replies":[]}') == []
    assert parse_batch_replies("") == []
    assert parse_batch_replies("not json") == []
    assert parse_batch_replies("<refuse/>") == []
    assert parse_batch_replies('{"replies":"nope"}') == []


def test_parse_batch_replies_ignores_unknown_and_missing_target_ids() -> None:
    raw = (
        '{"replies":['
        '{"message_id":"999","text":"stale id","quote":true},'
        '{"text":"missing id","quote":true},'
        '{"message_id":"1","text":"ok","quote":true}'
        "]}"
    )
    replies = parse_batch_replies(raw, max_replies=5)
    assert [reply.text for reply in replies] == ["stale id", "missing id", "ok"]


def test_parse_batch_replies_respects_max_replies_and_skips_empty_text() -> None:
    raw = (
        '{"replies":['
        '{"text":"a"},'
        '{"text":"   "},'
        '{"text":"b"},'
        '{"text":"c"},'
        "]}"
    )
    replies = parse_batch_replies(raw, max_replies=2)
    assert [reply.text for reply in replies] == ["a", "b"]


def test_extract_json_object_with_surrounding_text() -> None:
    parsed = extract_json_object('好的\n{"replies":[]}\n结束')
    assert parsed == {"replies": []}


def test_split_discord_text_preserves_content_and_limits_size() -> None:
    text = "line one\nline two\n" + ("x" * 2500) + "\nlast line"
    chunks = split_discord_text(text)
    assert len(chunks) >= 2
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks) == text


def test_split_discord_text_short_text_single_chunk() -> None:
    assert split_discord_text("hello\nworld") == ["hello\nworld"]
    assert split_discord_text("") == []


def test_build_batch_prompt_asks_for_one_plain_reply() -> None:
    prompt = build_batch_prompt(
        history_text="=== CHAT_HISTORY_BEGIN ===\n[a]\n=== CHAT_HISTORY_END ===",
        batch_payload=[
            {"message_id": "100", "user": "Alice/1", "content": "hi", "time": "12:00"}
        ],
        mention_parse=True,
        persona_name="tester",
        persona_mask="be nice",
    )
    assert "exactly ONE reply" in prompt
    rules_section = prompt.split("Rules:", 1)[1]
    assert '"replies"' not in prompt
    assert "message_id" not in rules_section
    assert "quote" not in rules_section
    assert "100" in prompt
    assert "tester" in prompt and "be nice" in prompt
    assert "<mention" in prompt
    assert "<refuse/>" in prompt


def test_build_batch_prompt_without_mention_rule() -> None:
    prompt = build_batch_prompt(
        history_text="(empty)",
        batch_payload=[],
        mention_parse=False,
    )
    assert "<mention" not in prompt


def test_parse_batch_reply_text_plain_text_passthrough() -> None:
    assert parse_batch_reply_text("直接回复\n第二行") == "直接回复\n第二行"
    assert parse_batch_reply_text('  "带引号的普通文本"  ') == '"带引号的普通文本"'


def test_parse_batch_reply_text_silence_inputs() -> None:
    assert parse_batch_reply_text("") == ""
    assert parse_batch_reply_text(None) == ""
    assert parse_batch_reply_text("<refuse/>") == ""
    assert parse_batch_reply_text('{"replies":[]}') == ""
    assert parse_batch_reply_text('{"other":"x"}') == ""


def test_parse_batch_reply_text_legacy_replies_takes_first_only() -> None:
    raw = (
        '{"replies":['
        '{"message_id":"100","text":"第一条回复","quote":true},'
        '{"message_id":"101","text":"第二条回复","quote":false}'
        "]}"
    )
    assert parse_batch_reply_text(raw) == "第一条回复"


def test_parse_batch_reply_text_single_text_object() -> None:
    assert parse_batch_reply_text('{"text":"单条回复"}') == "单条回复"
