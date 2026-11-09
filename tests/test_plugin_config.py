from __future__ import annotations

import math

from astrbot_plugin_astrbot_enhance_mode.plugin_config import parse_plugin_config


def test_parse_plugin_config_defaults() -> None:
    cfg = parse_plugin_config(None)
    assert cfg.group_history.enable is False
    assert cfg.active_reply.enable is False
    assert cfg.active_reply.mode == "probability"
    assert cfg.active_reply.model_choice_provider_id == ""
    assert math.isclose(cfg.active_reply.possibility, 0.1)
    assert cfg.group_history_enabled is False
    assert cfg.active_reply_enabled is False
    assert cfg.web_search.request_mode == "auto"
    assert cfg.web_search.base_url_override == ""


def test_probability_is_clamped_and_nan_falls_back() -> None:
    cfg_high = parse_plugin_config(
        {
            "group_features": {"react_mode_enable": True},
            "active_reply": {"enable": True, "possibility": 9},
        }
    )
    assert math.isclose(cfg_high.active_reply.possibility, 1.0)
    assert cfg_high.active_reply_enabled is True

    cfg_low = parse_plugin_config({"active_reply": {"possibility": -0.5}})
    assert math.isclose(cfg_low.active_reply.possibility, 0.0)

    cfg_nan = parse_plugin_config({"active_reply": {"possibility": "nan"}})
    assert math.isclose(cfg_nan.active_reply.possibility, 0.1)


def test_active_reply_mode_and_limits_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "active_reply": {
                "mode": "something_else",
                "model_stack_size": 0,
                "model_history_messages": -99,
                "model_choice_provider_id": "  provider-1  ",
                "whitelist": "a,b, c",
            },
            "global_settings": {
                "lru_cache": {"max_origins": 0},
                "timeouts": {"image_caption_sec": -1, "model_choice_sec": "0"},
            },
        }
    )

    assert cfg.active_reply.mode == "probability"
    assert cfg.active_reply.model_stack_size == 1
    assert cfg.active_reply.model_history_messages == 0
    assert cfg.active_reply.model_choice_provider_id == "provider-1"
    assert cfg.active_reply.whitelist == ["a", "b", "c"]
    assert cfg.global_settings.lru_cache.max_origins == 1
    assert math.isclose(cfg.global_settings.timeouts.image_caption_sec, 45.0)
    assert math.isclose(cfg.global_settings.timeouts.model_choice_sec, 45.0)


def test_web_search_request_mode_and_base_override_are_normalized() -> None:
    cfg = parse_plugin_config(
        {
            "web_search": {
                "request_mode": "RESPONSES",
                "base_url_override": "  https://example.com/custom/v1  ",
            }
        }
    )
    assert cfg.web_search.request_mode == "responses"
    assert cfg.web_search.base_url_override == "https://example.com/custom/v1"

    cfg_invalid = parse_plugin_config({"web_search": {"request_mode": "unknown"}})
    assert cfg_invalid.web_search.request_mode == "auto"


def test_batch_reply_defaults_are_disabled_and_documented_values() -> None:
    cfg = parse_plugin_config(None)
    assert cfg.batch_reply.enable is False
    assert cfg.batch_reply.round_seconds == 8.0
    assert cfg.batch_reply.silent_rounds == 3
    assert cfg.batch_reply.max_replies == 5
    assert cfg.batch_reply.max_pending == 50
    assert cfg.batch_reply.reply_interval_ms == 800
    assert cfg.batch_reply_enabled is False


def test_batch_reply_is_gated_by_react_mode() -> None:
    cfg = parse_plugin_config({"batch_reply": {"enable": True}})
    assert cfg.batch_reply.enable is True
    assert cfg.batch_reply_enabled is False

    cfg_react = parse_plugin_config(
        {
            "group_features": {"react_mode_enable": True},
            "batch_reply": {"enable": True},
        }
    )
    assert cfg_react.batch_reply_enabled is True


def test_batch_reply_values_are_clamped() -> None:
    cfg = parse_plugin_config(
        {
            "batch_reply": {
                "round_seconds": 0,
                "silent_rounds": 0,
                "max_replies": 99,
                "max_pending": 0,
                "reply_interval_ms": -5,
            }
        }
    )
    assert cfg.batch_reply.round_seconds == 1.0
    assert cfg.batch_reply.silent_rounds == 1
    assert cfg.batch_reply.max_replies == 5
    assert cfg.batch_reply.max_pending == 1
    assert cfg.batch_reply.reply_interval_ms == 0

    cfg_high = parse_plugin_config(
        {
            "batch_reply": {
                "round_seconds": 99999,
                "silent_rounds": 9999,
                "max_replies": 999,
                "max_pending": 99999,
                "reply_interval_ms": 999999,
            }
        }
    )
    assert cfg_high.batch_reply.round_seconds == 600.0
    assert cfg_high.batch_reply.silent_rounds == 100
    assert cfg_high.batch_reply.max_replies == 5
    assert cfg_high.batch_reply.max_pending == 500
    assert cfg_high.batch_reply.reply_interval_ms == 10000


def test_batch_reply_invalid_and_nan_values_fall_back() -> None:
    cfg = parse_plugin_config(
        {
            "batch_reply": {
                "enable": "true",
                "round_seconds": "nan",
                "silent_rounds": "abc",
                "max_replies": None,
            }
        }
    )
    assert cfg.batch_reply.enable is True
    assert cfg.batch_reply.round_seconds == 8.0
    assert cfg.batch_reply.silent_rounds == 3
    assert cfg.batch_reply.max_replies == 5
