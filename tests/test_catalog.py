"""Catalog fallback tests.

Contract under test:

- ``fetch_models`` is binary-first: when the ``auth-cloudflare`` executable
  is present and compatible, the catalog comes from ``catalog get --format
  json`` (that delegation path is covered by ``test_binary_discovery.py``).
- Without a binary, or on any binary catalog error, ``fetch_models`` returns
  the static ``FALLBACK_MODELS`` list - there is no direct in-process HTTP
  catalog discovery anymore (the Rust binary owns live fetching).
- Llama Guard (safety classifier) is excluded from the primary picker.
- DeepSeek V4 Flash is the sole default; GLM-5.3 Flash stays available but
  non-default.

No network: ``locate_auth_cloudflare_binary`` is forced to ``None`` so the
static-fallback branch is exercised in-process, never a live fetch.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from helpers import load_plugin

plugin = load_plugin()

ACCOUNT = "testacct0001"
SYNTHETIC_TOKEN = "cfut_test_catalog_only"  # never a real credential
DEEPSEEK = "@cf/deepseek-ai/deepseek-v4-flash-0731"
GLM_FLASH = "@cf/zai-org/glm-5.3-flash"
KIMI_CODE = "@cf/moonshotai/kimi-k2.7-code"
LLAMA_GUARD = "@cf/meta/llama-guard-3-8b"


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self._saved_env = dict(os.environ)
        os.environ[plugin.AUTH_ACCOUNT_ENV] = ACCOUNT
        os.environ[plugin.AUTH_TOKEN_ENV] = SYNTHETIC_TOKEN
        # The auth-cloudflare binary is installed on dev machines; these tests
        # assert the static-fallback path, so force the locator to None
        # (binary-first delegation has its own coverage in
        # test_binary_discovery.py).
        self._locator = mock.patch.object(
            plugin, "locate_auth_cloudflare_binary", return_value=None
        )
        self._locator.start()

    def tearDown(self):
        self._locator.stop()
        os.environ.clear()
        os.environ.update(self._saved_env)

    def test_fetch_models_returns_static_fallback_without_binary(self):
        result = plugin.cloudflare.fetch_models()
        self.assertEqual(result, list(plugin.FALLBACK_MODELS))
        self.assertIn(DEEPSEEK, result)
        self.assertIn(GLM_FLASH, result)
        self.assertIn(KIMI_CODE, result)

    def test_llama_guard_excluded_from_fallback(self):
        result = plugin.cloudflare.fetch_models()
        self.assertNotIn(LLAMA_GUARD, result)

    def test_llama_guard_not_in_fallback_models(self):
        self.assertNotIn(LLAMA_GUARD, plugin.FALLBACK_MODELS)
        self.assertNotIn(LLAMA_GUARD, plugin.cloudflare.fallback_models)

    def test_guard_fragment_is_filtered(self):
        self.assertIn("guard", plugin._NON_CHAT_FRAGMENTS)

    def test_fallback_returned_even_without_account_or_token(self):
        # The static fallback is credential-independent: no account or token
        # is required, and no request is ever made.
        os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
        os.environ.pop(plugin.ACCOUNT_ENV, None)
        os.environ.pop(plugin.AUTH_TOKEN_ENV, None)
        os.environ.pop(plugin.TOKEN_ENV, None)
        result = plugin.cloudflare.fetch_models()
        self.assertEqual(result, list(plugin.FALLBACK_MODELS))

    def test_deepseek_flash_is_the_only_default(self):
        self.assertEqual(plugin.DEFAULT_MODEL, DEEPSEEK)
        self.assertEqual(plugin.FALLBACK_MODELS[0], DEEPSEEK)
        self.assertEqual(plugin.cloudflare.default_aux_model, DEEPSEEK)

    def test_glm_flash_never_default(self):
        self.assertNotEqual(plugin.DEFAULT_MODEL, GLM_FLASH)
        self.assertNotEqual(plugin.FALLBACK_MODELS[0], GLM_FLASH)

    def test_fallback_list_is_the_result_without_binary(self):
        result = plugin.cloudflare.fetch_models()
        self.assertEqual(result, list(plugin.FALLBACK_MODELS))
        # The fallback list remains available to the picker, default first.
        self.assertGreaterEqual(len(plugin.FALLBACK_MODELS), 1)
        self.assertEqual(plugin.cloudflare.fallback_models, plugin.FALLBACK_MODELS)
        self.assertEqual(plugin.cloudflare.fallback_models[0], DEEPSEEK)


if __name__ == "__main__":
    unittest.main(verbosity=2)
