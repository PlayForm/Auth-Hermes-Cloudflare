"""URL construction tests (feedback 05/06, binding).

Contract under test:

- Inference base URL is exactly
  ``https://api.cloudflare.com/client/v4/accounts/<id>/ai/v1``.
- Catalog route is exactly
  ``https://api.cloudflare.com/client/v4/accounts/<id>/ai/models/search``
  (NOT OpenAI's ``/models``).
- Without an account ID the catalog URL is None and the inference URL uses
  the ``<ACCOUNT_ID>`` placeholder.
- ``CLOUDFLARE_BASE_URL`` is declared in ``env_vars`` so the stock setup
  wizard pre-fills its Base URL prompt from the account-derived URL (the
  ``*_BASE_URL`` suffix maps to ``ProviderConfig.base_url_env_var``).
"""

from __future__ import annotations

import os
import unittest

from helpers import load_plugin

plugin = load_plugin()

ACCOUNT = "testacct0001"
API_BASE = "https://api.cloudflare.com/client/v4"


class UrlTest(unittest.TestCase):
	def setUp(self):
		os.environ.pop(plugin.BASE_URL_ENV, None)
		self._saved = dict(os.environ)
		os.environ[plugin.AUTH_ACCOUNT_ENV] = ACCOUNT

	def tearDown(self):
		os.environ.clear()
		os.environ.update(self._saved)

	def test_inference_base_url_exact(self):
		self.assertEqual(
			plugin.inference_base_url(),
			f"{API_BASE}/accounts/{ACCOUNT}/ai/v1",
		)

	def test_inference_base_url_prefers_env_override(self):
		os.environ[plugin.BASE_URL_ENV] = "https://example.com/override/v1"
		self.assertEqual(
			plugin.inference_base_url(),
			"https://example.com/override/v1",
		)

	def test_inference_base_url_contains_account_id(self):
		url = plugin.inference_base_url()
		self.assertIn(f"/accounts/{ACCOUNT}/", url)

	def test_catalog_url_exact_route(self):
		self.assertEqual(
			plugin.catalog_url(),
			f"{API_BASE}/accounts/{ACCOUNT}/ai/models/search"
			"?format=openrouter&per_page=1000",
		)

	def test_catalog_uses_models_search_not_openai_models(self):
		url = plugin.catalog_url()
		path = url.split("?", 1)[0]
		self.assertTrue(path.endswith("/ai/models/search"))
		self.assertFalse(path.endswith("/models"))

	def test_catalog_url_none_without_account(self):
		os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
		os.environ.pop(plugin.ACCOUNT_ENV, None)
		self.assertIsNone(plugin.catalog_url())

	def test_inference_url_placeholder_without_account(self):
		os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
		os.environ.pop(plugin.ACCOUNT_ENV, None)
		self.assertEqual(
			plugin.inference_base_url(),
			f"{API_BASE}/accounts/<ACCOUNT_ID>/ai/v1",
		)

	def test_profile_base_url_property_matches(self):
		self.assertEqual(plugin.cloudflare.base_url, plugin.inference_base_url())

	def test_profile_models_url_property_matches(self):
		self.assertEqual(plugin.cloudflare.models_url, plugin.catalog_url())

	def test_models_url_empty_without_account(self):
		os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
		os.environ.pop(plugin.ACCOUNT_ENV, None)
		self.assertEqual(plugin.cloudflare.models_url, "")

	def test_base_url_env_var_declared(self):
		# Stock hermes-agent maps any *_BASE_URL env var in env_vars to
		# ProviderConfig.base_url_env_var, so the setup wizard pre-fills its
		# Base URL prompt from the account-derived URL (feedback 06 + HANDOFF).
		self.assertIn(plugin.BASE_URL_ENV, plugin.cloudflare.env_vars)
		self.assertTrue(plugin.BASE_URL_ENV.endswith("_BASE_URL"))
		# Legacy patched cores additionally carry a fixed_base_url flag; it
		# is set post-construction only when the base class declares it.
		if hasattr(plugin.ProviderProfile, "fixed_base_url"):
			self.assertTrue(plugin.cloudflare.fixed_base_url)

	def test_base_url_uses_client_v4_prefix(self):
		self.assertTrue(plugin.inference_base_url().startswith(API_BASE))


if __name__ == "__main__":
	unittest.main(verbosity=2)
