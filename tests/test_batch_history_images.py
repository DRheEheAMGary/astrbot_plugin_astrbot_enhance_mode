from __future__ import annotations

import pytest
from mcp import types as mcp_types

from astrbot.api.message_components import Image, Plain

from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import (
    BATCH_HISTORY_IMAGE_LIMIT,
    BATCH_PAYLOAD_EXTRA,
    build_batch_prompt,
)
from astrbot_plugin_astrbot_enhance_mode.main import Main
from test_batch_replies import _FakeEvent, _build_plugin, _config


def _image(url: str) -> Image:
    return Image(file=url, url=url)


def _text_of(results: list[mcp_types.CallToolResult]) -> str:
    parts: list[str] = []
    for result in results:
        for content in result.content:
            text = getattr(content, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _has_image(results: list[mcp_types.CallToolResult]) -> bool:
    return any(
        isinstance(content, mcp_types.ImageContent)
        for result in results
        for content in result.content
    )


async def _call_use_image(plugin, event, **kwargs) -> list[mcp_types.CallToolResult]:
    results: list[mcp_types.CallToolResult] = []
    async for result in plugin.use_image(event, **kwargs):
        results.append(result)
    assert results
    return results


def _setup_use_image_plugin():
    plugin = _build_plugin(_config(history_enable=True))

    async def _resolve_local(ref: str) -> str:
        return f"/tmp/{abs(hash(ref))}.png"

    plugin._resolve_image_ref_to_local_path = _resolve_local
    plugin._encode_image_file = lambda _path: ("ZmFrZQ==", "image/png")
    origin = "discord/GroupMessage/chan-1"
    plugin.runtime.image_message_registry[origin] = {
        "100": {"urls": ["https://cdn/a.png", "https://cdn/b.png"], "captions": {}},
        "101": {"urls": ["https://cdn/c.png", "https://cdn/d.png"], "captions": {}},
    }
    return plugin, origin


@pytest.mark.asyncio
async def test_record_message_appends_image_count_marker() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(
        message_id="100",
        text="看图",
        components=[
            Plain(text="看图 "),
            _image("https://cdn/one.png"),
            Plain(text=" "),
            _image("https://cdn/two.png"),
        ],
    )

    await Main._record_message(plugin, event, plugin._cfg())

    line = plugin.runtime.session_chats[event.unified_msg_origin][0]
    assert "#msg100" in line
    assert line.endswith("[images: 2]")
    registry = plugin.runtime.image_message_registry[event.unified_msg_origin]["100"]
    assert registry["urls"] == ["https://cdn/one.png", "https://cdn/two.png"]


@pytest.mark.asyncio
async def test_record_message_without_images_has_no_marker() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="纯文字")

    await Main._record_message(plugin, event, plugin._cfg())

    line = plugin.runtime.session_chats[event.unified_msg_origin][0]
    assert "[images:" not in line


@pytest.mark.asyncio
async def test_use_image_on_demand_attach_and_dedupe_in_batch() -> None:
    plugin, origin = _setup_use_image_plugin()
    event = _FakeEvent()
    event.set_extra(BATCH_PAYLOAD_EXTRA, [{"message_id": "100"}])

    first = await _call_use_image(
        plugin,
        event,
        message_id="100",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
    )
    assert _has_image(first)

    duplicate = await _call_use_image(
        plugin,
        event,
        message_id="100",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
    )
    assert not _has_image(duplicate)
    assert "already attached" in _text_of(duplicate)

    second = await _call_use_image(
        plugin,
        event,
        message_id="100",
        image_index=2,
        attach_to_model=True,
        write_to_history=False,
    )
    assert _has_image(second)


@pytest.mark.asyncio
async def test_use_image_on_demand_limit_in_batch() -> None:
    plugin, origin = _setup_use_image_plugin()
    event = _FakeEvent()
    event.set_extra(BATCH_PAYLOAD_EXTRA, [{"message_id": "100"}])

    for message_id, image_index in (("100", 1), ("100", 2), ("101", 1)):
        result = await _call_use_image(
            plugin,
            event,
            message_id=message_id,
            image_index=image_index,
            attach_to_model=True,
            write_to_history=False,
        )
        assert _has_image(result)

    over = await _call_use_image(
        plugin,
        event,
        message_id="101",
        image_index=2,
        attach_to_model=True,
        write_to_history=False,
    )
    assert not _has_image(over)
    assert "limit reached" in _text_of(over).lower()
    assert BATCH_HISTORY_IMAGE_LIMIT == 3


@pytest.mark.asyncio
async def test_use_image_failure_degrades_to_text_without_blocking() -> None:
    plugin, origin = _setup_use_image_plugin()
    event = _FakeEvent()
    event.set_extra(BATCH_PAYLOAD_EXTRA, [{"message_id": "100"}])

    missing = await _call_use_image(
        plugin,
        event,
        message_id="999",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
    )
    assert not _has_image(missing)
    assert "not found" in _text_of(missing)

    out_of_range = await _call_use_image(
        plugin,
        event,
        message_id="100",
        image_index=9,
        attach_to_model=True,
        write_to_history=False,
    )
    assert not _has_image(out_of_range)
    assert "out of range" in _text_of(out_of_range)

    # Failures did not consume the per-round budget.
    ok = await _call_use_image(
        plugin,
        event,
        message_id="100",
        image_index=1,
        attach_to_model=True,
        write_to_history=False,
    )
    assert _has_image(ok)


@pytest.mark.asyncio
async def test_use_image_without_batch_keeps_old_behavior() -> None:
    plugin, origin = _setup_use_image_plugin()
    event = _FakeEvent()  # no batch payload extra

    for _ in range(BATCH_HISTORY_IMAGE_LIMIT + 1):
        result = await _call_use_image(
            plugin,
            event,
            message_id="100",
            image_index=1,
            attach_to_model=True,
            write_to_history=False,
        )
        assert _has_image(result)


@pytest.mark.asyncio
async def test_batch_attach_without_caption_config_does_not_fail() -> None:
    plugin, origin = _setup_use_image_plugin()
    # History image captioning is disabled by default in this test config.
    event = _FakeEvent()
    event.set_extra(BATCH_PAYLOAD_EXTRA, [{"message_id": "100"}])

    result = await _call_use_image(
        plugin,
        event,
        message_id="100",
        image_index=1,
        attach_to_model=True,
        write_to_history=True,  # model forgot the recommended params
    )
    assert _has_image(result)


def test_batch_prompt_mentions_on_demand_history_images() -> None:
    prompt = build_batch_prompt(
        history_text="=== CHAT_HISTORY_BEGIN ===\n[a] #msg1: hi [images: 2]\n=== CHAT_HISTORY_END ===",
        batch_payload=[
            {"message_id": "2", "user": "A/1", "content": "text", "time": "12:00"}
        ],
        mention_parse=False,
        history_images_available=True,
        history_image_limit=BATCH_HISTORY_IMAGE_LIMIT,
    )
    assert "enhance_use_image" in prompt
    assert "[images: N]" in prompt
    assert "[Image]" in prompt
    assert "#msg<message_id>" in prompt
    assert f"at most {BATCH_HISTORY_IMAGE_LIMIT} images" in prompt
    assert "write_to_history=false" in prompt


def test_batch_prompt_omits_on_demand_rule_when_history_disabled() -> None:
    prompt = build_batch_prompt(
        history_text="(empty)",
        batch_payload=[],
        mention_parse=False,
        history_images_available=False,
    )
    assert "enhance_use_image" not in prompt


@pytest.mark.asyncio
async def test_make_batch_request_injects_history_image_guidance() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    plugin.runtime.session_chats[event.unified_msg_origin] = [
        "[Alice/1/12:00:00](member) #msg900: 看这张 [Image] [images: 1]"
    ]
    message = plugin._build_batch_message(event)

    cfg = plugin._cfg()
    await plugin._make_batch_request(event, [message], cfg)

    prompt = event.requests[0]["prompt"]
    assert "enhance_use_image" in prompt
    assert "images: N" in prompt


@pytest.mark.asyncio
async def test_make_batch_request_skips_guidance_without_history_images() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    plugin.runtime.session_chats[event.unified_msg_origin] = [
        "[Alice/1/12:00:00](member) #msg900: 纯文本旧消息"
    ]
    message = plugin._build_batch_message(event)

    await plugin._make_batch_request(event, [message], plugin._cfg())

    prompt = event.requests[0]["prompt"]
    assert "enhance_use_image" not in prompt


@pytest.mark.asyncio
async def test_make_batch_request_skips_guidance_for_current_batch_images_only() -> None:
    plugin = _build_plugin(_config(history_enable=True))
    event = _FakeEvent(message_id="100", text="hi")
    plugin.runtime.session_chats[event.unified_msg_origin] = [
        "[Alice/1/12:00:00](member) #msg100: 当前批次图 [Image] [images: 1]"
    ]
    message = plugin._build_batch_message(event)

    await plugin._make_batch_request(event, [message], plugin._cfg())

    prompt = event.requests[0]["prompt"]
    assert "enhance_use_image" not in prompt


@pytest.mark.asyncio
async def test_make_batch_request_skips_history_image_guidance_when_disabled() -> None:
    plugin = _build_plugin(_config(history_enable=False))
    event = _FakeEvent(message_id="100", text="hi")
    message = plugin._build_batch_message(event)

    await plugin._make_batch_request(event, [message], plugin._cfg())

    prompt = event.requests[0]["prompt"]
    assert "enhance_use_image" not in prompt
