"""Environment resolution tests.

Contract under test:

- The account ID and API token are read from the canonical
  ``CLOUDFLARE_ACCOUNT_ID`` / ``CLOUDFLARE_API_TOKEN`` variables.
- Whitespace-only values are treated as missing (``_env`` strips).
"""

from __future__ import annotations

import os
import unittest

from helpers import load_plugin

plugin = load_plugin()

SYNTHETIC_TOKEN = "cfut_test_environment_only"  # never a real credential


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        # Isolate the disk fallbacks api_token()/account_id() now consult
        # (the real ~/.hermes/.env and credential pool would otherwise leak
        # into "missing value" assertions). Restored in tearDown.
        self._saved_dotenv = plugin._dotenv_value
        self._saved_pool = plugin._pool_api_token
        plugin._dotenv_value = lambda key: None
        plugin._pool_api_token = lambda provider: None

    def tearDown(self):
        plugin._dotenv_value = self._saved_dotenv
        plugin._pool_api_token = self._saved_pool
        os.environ.clear()
        os.environ.update(self._saved)

    def test_all_whitespace_account_returns_none(self):
        os.environ[plugin.ACCOUNT_ENV] = "   "
        self.assertIsNone(plugin.account_id())

    def test_all_whitespace_token_returns_none(self):
        os.environ[plugin.TOKEN_ENV] = "  "
        self.assertIsNone(plugin.api_token())

    def test_empty_string_returns_none(self):
        os.environ[plugin.ACCOUNT_ENV] = ""
        self.assertIsNone(plugin.account_id())

    def test_missing_variables_return_none(self):
        os.environ.pop(plugin.ACCOUNT_ENV, None)
        os.environ.pop(plugin.TOKEN_ENV, None)
        self.assertIsNone(plugin.account_id())
        self.assertIsNone(plugin.api_token())

    def test_env_constant_names_are_exact(self):
        self.assertEqual(plugin.ACCOUNT_ENV, "CLOUDFLARE_ACCOUNT_ID")
        self.assertEqual(plugin.TOKEN_ENV, "CLOUDFLARE_API_TOKEN")

    def test_profile_env_vars(self):
        # The account id is deliberately NOT an env_vars entry: stock
        # _api_key_env_fields(env_vars) treats every non-URL var as an
        # api-key credential, so including it seeds the credential pool with
        # a fake key (the 401 source). It is read directly via account_id().
        profile = plugin.cloudflare
        self.assertIn(plugin.TOKEN_ENV, profile.env_vars)
        self.assertIn(plugin.BASE_URL_ENV, profile.env_vars)
        self.assertNotIn(plugin.ACCOUNT_ENV, profile.env_vars)


if __name__ == "__main__":
    unittest.main(verbosity=2)