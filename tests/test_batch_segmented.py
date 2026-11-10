from __future__ import annotations

import json
import math
import random

import pytest

from astrbot.api.message_components import Plain, Reply

from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import (
    BATCH_PAYLOAD_EXTRA,
    DEFAULT_SEGMENT_LOG_BASE,
    SegmentedReplyOptions,
    calc_segment_interval,
    parse_segmented_reply_options,
    segment_word_count,
    split_segment_text,
)
from test_batch_replies import (
    _FakeContext,
    _FakeEvent,
    _build_plugin,
    _config,
    _decorating_event,
)


# ---------------------------------------------------------------------------
# option parsing
# ---------------------------------------------------------------------------


def test_parse_segmented_reply_options_defaults_match_astrbot() -> None:
    options = parse_segmented_reply_options({})
    assert options.enable is False
    assert options.only_llm_result is False
    assert options.words_count_threshold == 1000
    assert options.split_mode == "regex"
    assert options.interval_method == "log"
    assert options.log_base == DEFAULT_SEGMENT_LOG_BASE
    assert options.split_words == ("。", "？", "！", "~", "…")


def test_parse_segmented_reply_options_normalizes_values() -> None:
    options = parse_segmented_reply_options(
        {
            "enable": "true",
            "only_llm_result": "true",
            "words_count_threshold": "200",
            "split_mode": "WORDS",
            "split_words": ["，", "。"],
            "content_cleanup_rule": r"\s+",
            "interval_method": "random",
            "interval": "0.1, 0.2",
            "log_base": 0,
        }
    )
    assert options.enable is True
    assert options.only_llm_result is True
    assert options.words_count_threshold == 200
    assert options.split_mode == "words"
    assert options.split_words == ("，", "。")
    assert options.content_cleanup_rule == r"\s+"
    assert options.interval_method == "random"
    assert (options.interval_min, options.interval_max) == (0.1, 0.2)
    assert options.log_base == DEFAULT_SEGMENT_LOG_BASE  # invalid base falls back


def test_parse_segmented_reply_options_invalid_mode_falls_back() -> None:
    options = parse_segmented_reply_options(
        {"split_mode": "unknown", "interval_method": "unknown", "interval": "bad"}
    )
    assert options.split_mode == "regex"
    assert options.interval_method == "log"
    assert (options.interval_min, options.interval_max) == (1.5, 3.5)


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------


def _newline_options(**overrides) -> SegmentedReplyOptions:
    payload = {
        "enable": True,
        "words_count_threshold": 1000,
        "split_mode": "regex",
        "regex": r".*?[\n]+|.+$",
    }
    payload.update(overrides)
    return parse_segmented_reply_options(payload)


def test_split_segment_text_regex_splits_on_newlines_and_strips() -> None:
    options = _newline_options()
    assert split_segment_text("第一段\n第二段\n", options) == ["第一段", "第二段"]
    assert split_segment_text("只有一段", options) == ["只有一段"]


def test_split_segment_text_respects_words_count_threshold() -> None:
    options = _newline_options(words_count_threshold=5)
    long_text = "123456\nline2"
    assert split_segment_text(long_text, options) == [long_text]


def test_split_segment_text_words_mode() -> None:
    options = parse_segmented_reply_options(
        {
            "enable": True,
            "split_mode": "words",
            "split_words": ["。", "？"],
            "words_count_threshold": 1000,
        }
    )
    assert split_segment_text("你好。再见。", options) == ["你好", "再见"]
    assert split_segment_text("没有分隔符", options) == ["没有分隔符"]


def test_split_segment_text_applies_content_cleanup_rule() -> None:
    options = _newline_options(content_cleanup_rule=r"\[[^\]]*\]")
    assert split_segment_text("[tag]你好\n[tag]再见", options) == ["你好", "再见"]


def test_split_segment_text_invalid_regex_falls_back() -> None:
    options = _newline_options(regex="([")
    # Falls back to AstrBot's default sentence regex.
    assert split_segment_text("a。b", options) == ["a。", "b"]


def test_split_segment_text_empty_text() -> None:
    options = _newline_options()
    assert split_segment_text("", options) == []
    assert split_segment_text("   ", options) == []


# ---------------------------------------------------------------------------
# intervals
# ---------------------------------------------------------------------------


def test_segment_word_count_matches_astrbot_rule() -> None:
    assert segment_word_count("hello world") == 2
    assert segment_word_count("你好世界") == 4
    assert segment_word_count("你好 world") == 7  # non-ascii path counts alnum chars


def test_calc_segment_interval_log_method() -> None:
    options = parse_segmented_reply_options(
        {"enable": True, "interval_method": "log", "log_base": 2.6}
    )
    expected = math.log(2 + 1, 2.6)
    value = calc_segment_interval("hello world", options, rng=random.Random(0))
    assert expected <= value <= expected + 0.5
    # Non-Plain component uses the fixed 1..1.75 range.
    value_non_plain = calc_segment_interval(None, options, rng=random.Random(0))
    assert 1.0 <= value_non_plain <= 1.75


def test_calc_segment_interval_random_method() -> None:
    options = parse_segmented_reply_options(
        {"enable": True, "interval_method": "random", "interval": "0.2,0.4"}
    )
    value = calc_segment_interval("hello", options, rng=random.Random(0))
    assert 0.2 <= value <= 0.4


# ---------------------------------------------------------------------------
# decorating integration
# ---------------------------------------------------------------------------


def _segmented_context(**overrides) -> _FakeContext:
    config = {
        "enable": True,
        "only_llm_result": False,
        "words_count_threshold": 1000,
        "split_mode": "regex",
        "regex": r".*?[\n]+|.+$",
        "split_words": ["。", "？", "！", "~", "…"],
        "content_cleanup_rule": "",
        "interval_method": "random",
        "interval": "0,0",
        "log_base": 2.6,
    }
    config.update(overrides)
    return _FakeContext({"segmented_reply": config})


def _payload(message_ids: list[str]) -> list[dict[str, str]]:
    return [
        {"message_id": message_id, "user": "A/1", "content": "msg", "time": "12:00"}
        for message_id in message_ids
    ]


@pytest.mark.asyncio
async def test_decorating_splits_reply_by_segmented_reply_config() -> None:
    plugin = _build_plugin(_config(history_enable=True), context=_segmented_context())
    payload = _payload(["100"])
    raw = json.dumps(
        {"replies": [{"message_id": "100", "text": "第一段\n第二段", "quote": True}]},
        ensure_ascii=False,
    )
    event = _decorating_event(raw, payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 2
    first, second = event.sent
    assert not any(isinstance(c, Reply) for c in first)
    assert not any(isinstance(c, Reply) for c in second)
    assert any(isinstance(c, Plain) and c.text == "第一段" for c in first)
    assert any(isinstance(c, Plain) and c.text == "第二段" for c in second)


@pytest.mark.asyncio
async def test_decorating_segmented_reply_uses_only_first_legacy_reply() -> None:
    plugin = _build_plugin(
        _config(history_enable=True), context=_segmented_context()
    )
    payload = _payload(["100", "101"])
    raw = json.dumps(
        {
            "replies": [
                {"message_id": "100", "text": "A1\nA2", "quote": True},
                {"message_id": "101", "text": "B1", "quote": True},
            ]
        },
        ensure_ascii=False,
    )
    event = _decorating_event(raw, payload, cached={100, 101})

    await plugin.parse_tags(event)

    # Legacy list: only the first entry is used, then segmented into A1/A2.
    assert len(event.sent) == 2
    texts = [
        "".join(c.text for c in chain if isinstance(c, Plain)) for chain in event.sent
    ]
    assert texts == ["A1", "A2"]
    for chain in event.sent:
        assert not any(isinstance(c, Reply) for c in chain)


@pytest.mark.asyncio
async def test_decorating_segmented_reply_with_plain_text_output() -> None:
    plugin = _build_plugin(_config(history_enable=True), context=_segmented_context())
    payload = _payload(["100"])
    event = _decorating_event("第一段\n第二段", payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 2
    texts = [
        "".join(c.text for c in chain if isinstance(c, Plain)) for chain in event.sent
    ]
    assert texts == ["第一段", "第二段"]
    for chain in event.sent:
        assert not any(isinstance(c, Reply) for c in chain)


@pytest.mark.asyncio
async def test_decorating_segmented_reply_disabled_sends_single_message() -> None:
    plugin = _build_plugin(
        _config(history_enable=True),
        context=_FakeContext({"segmented_reply": {"enable": False}}),
    )
    payload = _payload(["100"])
    raw = json.dumps(
        {"replies": [{"message_id": "100", "text": "第一段\n第二段", "quote": True}]},
        ensure_ascii=False,
    )
    event = _decorating_event(raw, payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 1
    assert any(
        isinstance(c, Plain) and c.text == "第一段\n第二段" for c in event.sent[0]
    )


@pytest.mark.asyncio
async def test_decorating_segmented_reply_respects_only_llm_result() -> None:
    context = _segmented_context(only_llm_result=True)
    plugin = _build_plugin(_config(history_enable=True), context=context)
    payload = _payload(["100"])
    raw = json.dumps(
        {"replies": [{"message_id": "100", "text": "第一段\n第二段", "quote": True}]},
        ensure_ascii=False,
    )
    # Fake MessageEventResult defaults to GENERAL_RESULT -> not a model result.
    event = _decorating_event(raw, payload)

    await plugin.parse_tags(event)

    assert len(event.sent) == 1


@pytest.mark.asyncio
async def test_decorating_segmented_reply_skips_excluded_platform() -> None:
    class _DingTalkEvent(_FakeEvent):
        def get_platform_name(self) -> str:
            return "dingtalk"

    plugin = _build_plugin(_config(history_enable=True), context=_segmented_context())
    payload = _payload(["100"])
    raw = json.dumps(
        {"replies": [{"message_id": "100", "text": "第一段\n第二段", "quote": True}]},
        ensure_ascii=False,
    )
    event = _decorating_event(raw, payload)
    event.__class__ = _DingTalkEvent

    await plugin.parse_tags(event)

    assert len(event.sent) == 1


@pytest.mark.asyncio
async def test_decorating_segmented_reply_keeps_transport_split_for_long_text() -> None:
    plugin = _build_plugin(_config(history_enable=True), context=_segmented_context())
    payload = _payload(["100"])
    long_text = "x" * 2500
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
