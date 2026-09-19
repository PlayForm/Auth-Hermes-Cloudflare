"""Reasoning-effort wire integration tests.

Contract under test:

- Cloudflare's OpenAI-compatible chat-completions schema documents
  ``reasoning_effort`` as the enum low|medium|high (per-model docs, e.g.
  deepseek-v4-flash-0731) - narrower than Hermes' internal ladder and than the
  generic OpenAI-compat wire set (none..max).
- ``CloudflareProfile.build_api_kwargs_extras`` clamps Hermes' effort onto
  that set (nearest WEAKER, never escalating) and emits ``reasoning_effort``
  top-level; disabled / missing / "none" efforts OMIT the field entirely
  (Cloudflare has no "none" level and no ``thinking`` toggle on this surface).
- ``supported_reasoning_efforts`` is tri-state: the documented set for plugin
  chat models, ``()`` for everything else (never send reasoning params to an
  endpoint that does not document them).
"""

from __future__ import annotations

import unittest

from helpers import load_plugin

plugin = load_plugin()

CHAT_MODEL = "@cf/deepseek-ai/deepseek-v4-flash-0731"
GUARD_MODEL = "@cf/meta/llama-guard-3-8b"
UNKNOWN_MODEL = "@cf/some-vendor/brand-new-model"


class SupportedReasoningEffortsTest(unittest.TestCase):
    """Tri-state vocabulary contract."""

    def test_chat_model_declares_cloudflare_set(self):
        self.assertEqual(
            plugin.CloudflareProfile().supported_reasoning_efforts(CHAT_MODEL),
            ("low", "medium", "high"),
        )

    def test_other_primary_models_share_the_set(self):
        for mid in (
            "@cf/deepseek-ai/deepseek-v4-pro-0813",
            "@cf/moonshotai/kimi-k2.7-code",
            "@cf/zai-org/glm-5.3-flash",
        ):
            self.assertEqual(
                plugin.CloudflareProfile().supported_reasoning_efforts(mid),
                ("low", "medium", "high"),
            )

    def test_safety_model_accepts_no_reasoning_params(self):
        self.assertEqual(
            plugin.CloudflareProfile().supported_reasoning_efforts(GUARD_MODEL),
            (),
        )

    def test_unknown_model_accepts_no_reasoning_params(self):
        self.assertEqual(
            plugin.CloudflareProfile().supported_reasoning_efforts(UNKNOWN_MODEL),
            (),
        )

    def test_none_model_accepts_no_reasoning_params(self):
        self.assertEqual(plugin.CloudflareProfile().supported_reasoning_efforts(None), ())


class BuildApiKwargsExtrasTest(unittest.TestCase):
    """Wire clamping contract."""

    def _emit(self, model, reasoning_config):
        return plugin.CloudflareProfile().build_api_kwargs_extras(
            reasoning_config=reasoning_config, model=model
        )

    def test_effort_clamps_onto_documented_set(self):
        profile = plugin.CloudflareProfile()
        cases = {
            # Hermes tier -> Cloudflare wire value
            "ultra": "high",
            "max": "high",
            "xhigh": "high",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "minimal": "low",
        }
        for hermes_tier, wire in cases.items():
            extra, top = profile.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": hermes_tier},
                model=CHAT_MODEL,
            )
            self.assertEqual(extra, {}, f"extra_body must stay empty for {hermes_tier}")
            self.assertEqual(top, {"reasoning_effort": wire}, f"tier {hermes_tier}")

    def test_none_effort_omits_the_field(self):
        extra, top = self._emit(CHAT_MODEL, {"enabled": True, "effort": "none"})
        self.assertEqual((extra, top), ({}, {}))

    def test_disabled_reasoning_omits_the_field(self):
        extra, top = self._emit(CHAT_MODEL, {"enabled": False, "effort": "high"})
        self.assertEqual((extra, top), ({}, {}))

    def test_missing_effort_omits_the_field(self):
        extra, top = self._emit(CHAT_MODEL, {"enabled": True})
        self.assertEqual((extra, top), ({}, {}))
        extra, top = self._emit(CHAT_MODEL, None)
        self.assertEqual((extra, top), ({}, {}))

    def test_non_chat_model_never_receives_reasoning_params(self):
        for cfg in (
            {"enabled": True, "effort": "ultra"},
            {"enabled": True, "effort": "medium"},
            {"enabled": False, "effort": "high"},
        ):
            extra, top = self._emit(GUARD_MODEL, cfg)
            self.assertEqual((extra, top), ({}, {}))


class ClampEffortFallbackTest(unittest.TestCase):
    """The local ladder fallback (used when agent.reasoning_effort is not
    importable) must match the canonical clamp semantics."""

    def test_fallback_clamps_nearest_weaker(self):
        cases = {
            "ultra": "high",
            "max": "high",
            "xhigh": "high",
            "high": "high",
            "medium": "medium",
            "minimal": "low",
            "low": "low",
        }
        for hermes_tier, wire in cases.items():
            self.assertEqual(
                plugin._clamp_effort(hermes_tier, ("low", "medium", "high")),
                wire,
                f"tier {hermes_tier}",
            )

    def test_fallback_passes_bespoke_names_through(self):
        self.assertEqual(plugin._clamp_effort("balanced", ("low", "medium", "high")), "balanced")


if __name__ == "__main__":
    unittest.main()