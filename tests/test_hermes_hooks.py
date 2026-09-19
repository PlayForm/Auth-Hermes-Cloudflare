"""Deep Hermes integration tests - llm_request middleware + error hooks.

Contract under test (pure callbacks - no PluginManager, no live calls):

- ``_cloudflare_llm_request_middleware`` rewrites Cloudflare calls onto the
  documented OpenAI-compatible schema: ``max_tokens`` → ``max_completion_tokens``
  (Cloudflare marks max_tokens deprecated), ``reasoning_effort`` clamped to the
  low|medium|high enum, ``temperature`` clamped to the 0..2 range. Non-Cloudflare
  calls and unchanged requests pass through untouched (empty result → the
  middleware chain keeps the original payload).
- ``_cloudflare_api_error_classification`` maps Cloudflare status codes onto
  Hermes' FailoverReason vocabulary; non-Cloudflare calls return None so the
  default classifier applies.
"""

from __future__ import annotations

import unittest

from helpers import load_plugin

plugin = load_plugin()

CLOUDFLARE_PROVIDER = "auth-cloudflare-workers-ai"


class LlmRequestMiddlewareTest(unittest.TestCase):
    """The outgoing-request rewrite (runs before every Cloudflare API call)."""

    def _rewrite(self, request, provider=CLOUDFLARE_PROVIDER, base_url=""):
        return plugin._cloudflare_llm_request_middleware(
            {"request": dict(request)}, provider=provider, base_url=base_url
        )

    def test_max_tokens_becomes_max_completion_tokens(self):
        result = self._rewrite({"model": "@cf/deepseek-ai/deepseek-v4-flash-0731", "max_tokens": 2048})
        request = result["request"]
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["max_completion_tokens"], 2048)
        self.assertEqual(request["model"], "@cf/deepseek-ai/deepseek-v4-flash-0731")
        self.assertEqual(result["source"], "auth-hermes-cloudflare")

    def test_effort_clamps_onto_cloudflare_enum(self):
        result = self._rewrite({"reasoning_effort": "max"})
        self.assertEqual(result["request"]["reasoning_effort"], "high")

    def test_valid_effort_passes_through_unchanged(self):
        result = self._rewrite({"reasoning_effort": "medium"})
        self.assertEqual(result, {})
        result = self._rewrite({"reasoning_effort": "high"})
        self.assertEqual(result, {})

    def test_temperature_clamped_to_documented_range(self):
        result = self._rewrite({"temperature": 2.5})
        self.assertEqual(result["request"]["temperature"], 2.0)
        result = self._rewrite({"temperature": -1})
        self.assertEqual(result["request"]["temperature"], 0.0)
        result = self._rewrite({"temperature": 0.5})
        self.assertEqual(result, {})

    def test_unchanged_request_returns_empty(self):
        self.assertEqual(self._rewrite({"model": "@cf/deepseek-ai/deepseek-v4-flash-0731"}), {})

    def test_non_cloudflare_call_untouched(self):
        result = self._rewrite({"max_tokens": 2048, "reasoning_effort": "max"}, provider="openrouter")
        self.assertEqual(result, {})

    def test_cloudflare_custom_provider_and_host_match(self):
        result = self._rewrite(
            {"max_tokens": 512}, provider="cloudflare", base_url="https://api.cloudflare.com/client/v4/x"
        )
        self.assertIn("max_completion_tokens", result["request"])
        result = self._rewrite(
            {"max_tokens": 512}, provider="custom", base_url="https://api.cloudflare.com/..."
        )
        self.assertIn("max_completion_tokens", result["request"])


class ApiErrorClassificationTest(unittest.TestCase):
    """Cloudflare status codes → Hermes FailoverReason vocabulary."""

    def _classify(self, status_code, provider=CLOUDFLARE_PROVIDER):
        return plugin._cloudflare_api_error_classification(
            provider=provider, model="@cf/deepseek-ai/deepseek-v4-flash-0731", status_code=status_code
        )

    def test_rate_limit_maps_to_retryable_rotate(self):
        result = self._classify(429)
        self.assertEqual(result["reason"], "rate_limit")
        self.assertTrue(result["retryable"])
        self.assertTrue(result["should_rotate_credential"])

    def test_overload_maps_to_backoff(self):
        for status in (503, 529):
            result = self._classify(status)
            self.assertEqual(result["reason"], "overloaded", f"status {status}")
            self.assertTrue(result["retryable"])

    def test_server_error_maps_to_retry(self):
        for status in (500, 502):
            result = self._classify(status)
            self.assertEqual(result["reason"], "server_error", f"status {status}")
            self.assertTrue(result["retryable"])

    def test_auth_maps_to_rotate(self):
        for status in (401, 403):
            result = self._classify(status)
            self.assertEqual(result["reason"], "auth", f"status {status}")
            self.assertTrue(result["should_rotate_credential"])

    def test_non_cloudflare_call_returns_none(self):
        self.assertIsNone(self._classify(429, provider="openrouter"))
        self.assertIsNone(self._classify(429, provider="custom", ))


if __name__ == "__main__":
    unittest.main()