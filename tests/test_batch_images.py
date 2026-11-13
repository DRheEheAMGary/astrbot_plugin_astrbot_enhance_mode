from __future__ import annotations

import pytest

from astrbot.api.message_components import Image, Plain, Reply

from astrbot_plugin_astrbot_enhance_mode.batch_scheduler import (
    BATCH_MAX_IMAGES,
    BATCH_PAYLOAD_EXTRA,
    BatchMessage,
    build_batch_prompt,
)
from test_batch_replies import _FakeEvent, _build_plugin, _config


def _image(url: str) -> Image:
    return Image(file=url, url=url)


def _event_with(
    components: list,
    *,
    message_id: str = "100",
    text: str = "",
) -> _FakeEvent:
    event = _FakeEvent(
        message_id=message_id,
        text=text,
        components=components,
    )
    return event


async def _batch_request(plugin, event, batch, cfg):
    return await plugin._make_batch_request(event, batch, cfg)


def _patch_image_resolver(plugin, *, failing: set[str] | None = None):
    failing = failing or set()

    async def _resolve(raw_ref: str) -> str:
        if str(raw_ref) in failing:
            return ""
        return f"local:{raw_ref}"

    plugin._resolve_batch_image_ref = _resolve


@pytest.mark.asyncio
async def test_batch_images_numbered_globally_and_ordered() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()
    _patch_image_resolver(plugin)

    events = [
        _event_with(
            [Plain(text="看这个 "), _image("https://cdn.example/a.png")],
            message_id="100",
            text="看这个",
        ),
        _event_with([Plain(text="没图")], message_id="101", text="没图"),
        _event_with(
            [
                Plain(text="两张 "),
                _image("https://cdn.example/b.png"),
                Plain(text=" "),
                _image("https://cdn.example/c.png"),
            ],
            message_id="102",
            text="两张",
        ),
    ]
    batch = [plugin._build_batch_message(event) for event in events]
    assert [message.images for message in batch] == [
        ["https://cdn.example/a.png"],
        [],
        ["https://cdn.example/b.png", "https://cdn.example/c.png"],
    ]

    request = await _batch_request(plugin, events[0], batch, cfg)

    kwargs = events[0].requests[0]
    assert kwargs["image_urls"] == [
        "local:https://cdn.example/a.png",
        "local:https://cdn.example/b.png",
        "local:https://cdn.example/c.png",
    ]
    prompt = kwargs["prompt"]
    assert "[Image #1]" in prompt
    assert "[Image #2]" in prompt
    assert "[Image #3]" in prompt
    assert prompt.index("[Image #1]") < prompt.index("[Image #2]") < prompt.index(
        "[Image #3]"
    )
    assert "This request attaches 3 image(s)" in prompt
    assert request is not None

    # History payload stays human readable and never exposes refs/numbers.
    payload = events[0].get_extra(BATCH_PAYLOAD_EXTRA)
    assert [item["content"].count("[Image]") for item in payload] == [1, 0, 2]
    assert not any("#" in item["content"] for item in payload)
    assert not any("cdn.example" in item["content"] for item in payload)


@pytest.mark.asyncio
async def test_batch_images_failed_ref_keeps_markers_and_no_gap() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()
    _patch_image_resolver(plugin, failing={"https://cdn.example/broken.png"})

    events = [
        _event_with(
            [_image("https://cdn.example/broken.png")], message_id="100", text=""
        ),
        _event_with(
            [_image("https://cdn.example/ok.png")], message_id="101", text=""
        ),
        _event_with(
            [
                _image("https://cdn.example/x.png"),
                _image("https://cdn.example/broken.png"),
            ],
            message_id="102",
            text="",
        ),
    ]
    batch = [plugin._build_batch_message(event) for event in events]

    await _batch_request(plugin, events[0], batch, cfg)

    kwargs = events[0].requests[0]
    assert kwargs["image_urls"] == [
        "local:https://cdn.example/ok.png",
        "local:https://cdn.example/x.png",
    ]
    prompt = kwargs["prompt"]
    assert "[Image: 获取失败]" in prompt
    assert "[Image #1]" in prompt and "[Image #2]" in prompt
    assert "[Image #3]" not in prompt


@pytest.mark.asyncio
async def test_batch_images_over_limit_marked_not_silently_dropped() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()
    _patch_image_resolver(plugin)

    urls = [f"https://cdn.example/{index}.png" for index in range(BATCH_MAX_IMAGES + 2)]
    event = _event_with([_image(url) for url in urls], message_id="100", text="")
    batch = [plugin._build_batch_message(event)]

    await _batch_request(plugin, event, batch, cfg)

    kwargs = event.requests[0]
    assert len(kwargs["image_urls"]) == BATCH_MAX_IMAGES
    assert kwargs["image_urls"][0] == "local:https://cdn.example/0.png"
    prompt = kwargs["prompt"]
    assert f"[Image #{BATCH_MAX_IMAGES}]" in prompt
    assert f"[Image #{BATCH_MAX_IMAGES + 1}]" not in prompt
    assert prompt.count("[Image: 超出单批上限，未传给模型]") == 2


@pytest.mark.asyncio
async def test_batch_images_duplicate_ref_reuses_number() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()
    _patch_image_resolver(plugin)

    event = _event_with(
        [
            _image("https://cdn.example/same.png"),
            Plain(text=" and "),
            _image("https://cdn.example/same.png"),
        ],
        message_id="100",
        text="",
    )
    batch = [plugin._build_batch_message(event)]

    await _batch_request(plugin, event, batch, cfg)

    kwargs = event.requests[0]
    assert kwargs["image_urls"] == ["local:https://cdn.example/same.png"]
    prompt = kwargs["prompt"]
    assert prompt.count("[Image #1]") == 2
    assert "[Image #2]" not in prompt


@pytest.mark.asyncio
async def test_batch_text_only_request_has_no_image_urls() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()
    _patch_image_resolver(plugin)

    event = _event_with([Plain(text="纯文本")], message_id="100", text="纯文本")
    batch = [plugin._build_batch_message(event)]

    await _batch_request(plugin, event, batch, cfg)

    kwargs = event.requests[0]
    assert "image_urls" not in kwargs
    assert "[Image #" not in kwargs["prompt"]
    assert "This request attaches" not in kwargs["prompt"]


@pytest.mark.asyncio
async def test_batch_quoted_message_images_are_collected() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()
    _patch_image_resolver(plugin)

    quoted = Reply(
        id="9",
        sender_nickname="Bob",
        message_str="quoted text",
        chain=[Plain(text="quoted text"), _image("https://cdn.example/quote.png")],
    )
    event = _event_with(
        [quoted, Plain(text=" look")], message_id="100", text="look"
    )
    message = plugin._build_batch_message(event)
    assert message.images == ["https://cdn.example/quote.png"]
    assert "[Image]" in message.content

    await _batch_request(plugin, event, [message], cfg)

    kwargs = event.requests[0]
    assert kwargs["image_urls"] == ["local:https://cdn.example/quote.png"]
    assert "[Image #1]" in kwargs["prompt"]


def test_batch_message_payload_does_not_expose_images() -> None:
    message = BatchMessage(
        message_id="1",
        user="A/1",
        content="hi [Image]",
        time="12:00",
        images=["https://cdn.example/secret.png"],
    )
    payload = message.to_payload()
    assert payload == {
        "message_id": "1",
        "user": "A/1",
        "content": "hi [Image]",
        "time": "12:00",
    }


def test_build_batch_prompt_image_contract_only_when_images() -> None:
    with_images = build_batch_prompt(
        history_text="(empty)",
        batch_payload=[{"message_id": "1", "user": "A/1", "content": "[Image #1]", "time": "12:00"}],
        mention_parse=False,
        image_count=1,
    )
    assert "This request attaches 1 image(s)" in with_images
    assert "[Image #k]" in with_images

    without_images = build_batch_prompt(
        history_text="(empty)",
        batch_payload=[{"message_id": "1", "user": "A/1", "content": "hi", "time": "12:00"}],
        mention_parse=False,
    )
    assert "This request attaches" not in without_images


@pytest.mark.asyncio
async def test_resolve_batch_image_ref_formats() -> None:
    plugin = _build_plugin(_config())

    async def _fake_local(image_ref: str) -> str:
        if image_ref == "https://cdn.example/ok.png":
            return "/tmp/ok.png"
        if image_ref == "broken://x":
            raise RuntimeError("boom")
        return ""

    plugin._resolve_image_ref_to_local_path = _fake_local

    assert (
        await plugin._resolve_batch_image_ref("https://cdn.example/ok.png")
        == "/tmp/ok.png"
    )
    assert (
        await plugin._resolve_batch_image_ref("base64://QUJD")
        == "base64://QUJD"
    )
    assert (
        await plugin._resolve_batch_image_ref("data:image/png;base64,QUJD")
        == "base64://QUJD"
    )
    assert await plugin._resolve_batch_image_ref("broken://x") == ""
    assert await plugin._resolve_batch_image_ref("") == ""


@pytest.mark.asyncio
async def test_batch_images_resolution_failure_marks_and_continues() -> None:
    plugin = _build_plugin(_config())
    cfg = plugin._cfg()

    async def _fake_local(image_ref: str) -> str:
        if "bad" in image_ref:
            raise RuntimeError("download failed")
        return f"local:{image_ref}"

    plugin._resolve_image_ref_to_local_path = _fake_local

    event = _event_with(
        [
            _image("https://cdn.example/bad.png"),
            _image("https://cdn.example/good.png"),
        ],
        message_id="100",
        text="",
    )
    batch = [plugin._build_batch_message(event)]

    await _batch_request(plugin, event, batch, cfg)

    kwargs = event.requests[0]
    assert kwargs["image_urls"] == ["local:https://cdn.example/good.png"]
    assert "[Image: 获取失败]" in kwargs["prompt"]
    assert "[Image #1]" in kwargs["prompt"]
