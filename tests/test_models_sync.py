"""models sync mapping tests - the pure Hermes model_overrides mapper.

Contract under test:

- ``_model_overrides_from_records`` maps `auth-cloudflare models sync` records
  onto ``model_overrides.<provider>.<model>.<field>`` under BOTH provider keys
  the runtime resolves (plugin id + custom-provider id), using only the
  override fields Hermes' schema reads (context_window, supports_tools,
  supports_reasoning, model_family).
- Dotted model ids (qwen3.8-27b) survive as literal dict keys - they are never
  split on '.' (the config write path is a dict merge, not a dotted-key CLI).
- Records with no usable fields are skipped; non-eligible models (safety) are
  absent from the binary output and thus never claimed.
"""

from __future__ import annotations

import unittest

from helpers import load_plugin

plugin = load_plugin()

FLASH = "@cf/deepseek-ai/deepseek-v4-flash-0731"
GUARD = "@cf/meta/llama-guard-3-8b"
QWEN = "@cf/qwen/qwen3.8-27b"


class ModelOverridesMappingTest(unittest.TestCase):
    """The pure record → model_overrides mapper."""

    def test_maps_capability_records_to_both_provider_keys(self):
        records = [
            {
                "id": FLASH,
                "context_window": 1048576,
                "tool_call": True,
                "reasoning": True,
                "reasoning_efforts": ["low", "medium", "high"],
                "model_family": "deepseek-flash",
            },
            {"id": QWEN, "tool_call": True, "model_family": "qwen3"},
            {"id": GUARD, "context_window": 8000},
        ]
        overrides = plugin._model_overrides_from_records(records)
        self.assertEqual(set(overrides), {"auth-cloudflare-workers-ai", "cloudflare"})
        flash = overrides["auth-cloudflare-workers-ai"][FLASH]
        self.assertEqual(
            flash,
            {
                "context_window": 1048576,
                "supports_tools": True,
                "supports_reasoning": True,
                "model_family": "deepseek-flash",
            },
        )
        # Both provider keys carry the same entry.
        self.assertEqual(overrides["cloudflare"][FLASH], flash)

    def test_dotted_model_id_survives_as_literal_key(self):
        overrides = plugin._model_overrides_from_records(
            [{"id": QWEN, "tool_call": True, "model_family": "qwen3"}]
        )
        self.assertIn(QWEN, overrides["cloudflare"])
        self.assertEqual(overrides["cloudflare"][QWEN], {"supports_tools": True, "model_family": "qwen3"})

    def test_skips_records_without_usable_fields(self):
        overrides = plugin._model_overrides_from_records(
            [{"id": "@cf/vendor/brand-new"}, {"id": 5}, {"not_a_record": True}]
        )
        self.assertEqual(overrides, {})

    def test_reasoning_efforts_are_not_carried_into_config(self):
        overrides = plugin._model_overrides_from_records(
            [{"id": FLASH, "reasoning": True, "reasoning_efforts": ["low", "medium", "high"]}]
        )
        entry = overrides["cloudflare"][FLASH]
        self.assertEqual(entry, {"supports_reasoning": True})
        self.assertNotIn("reasoning_efforts", entry)

    def test_context_window_must_be_positive_int(self):
        for bad in (0, -1, "1048576", None, 1.5):
            overrides = plugin._model_overrides_from_records(
                [{"id": FLASH, "context_window": bad, "tool_call": True}]
            )
            entry = overrides["auth-cloudflare-workers-ai"][FLASH]
            self.assertNotIn("context_window", entry, f"context {bad!r}")

    def test_supports_vision_is_mapped(self):
        vision = "@cf/meta/llama-3.2-11b-vision-instruct"
        overrides = plugin._model_overrides_from_records(
            [{"id": vision, "supports_vision": True, "tool_call": True}]
        )
        entry = overrides["cloudflare"][vision]
        self.assertEqual(entry, {"supports_tools": True, "supports_vision": True})
        # No supports_vision claim when the record does not carry it.
        overrides = plugin._model_overrides_from_records([{"id": FLASH, "tool_call": True}])
        self.assertNotIn("supports_vision", overrides["cloudflare"][FLASH])


class VisionDefaultTest(unittest.TestCase):
    """The auxiliary vision hook - vision calls must never hit the text-only
    main model."""

    def test_default_vision_model_is_the_cloudflare_vision_model(self):
        self.assertEqual(
            plugin.cloudflare.default_vision_model(),
            "@cf/meta/llama-3.2-11b-vision-instruct",
        )


class ModelsSyncRouterTest(unittest.TestCase):
    """The command router's malformed-input paths (no binary involved)."""

    def test_bare_models_is_a_usage_error(self):
        result = plugin.cloudflare_command("models")
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("models sync", result["error"])

    def test_unknown_models_subcommand_is_a_usage_error(self):
        result = plugin.cloudflare_command("models fly")
        self.assertEqual(result["exit_code"], 2)

    def test_empty_models_is_a_usage_error(self):
        result = plugin.cloudflare_command("models")
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()