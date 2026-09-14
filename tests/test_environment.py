"""Environment resolution tests.

Contract under test:

- Canonical ``AUTH_CLOUDFLARE_*`` variables override legacy ``CLOUDFLARE_*``
  variables when both are set.
- Legacy variables are used when no canonical value is present.
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

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)

    def test_auth_account_overrides_legacy(self):
        os.environ[plugin.AUTH_ACCOUNT_ENV] = "auth-acct-123"
        os.environ[plugin.ACCOUNT_ENV] = "legacy-acct-456"
        self.assertEqual(plugin.account_id(), "auth-acct-123")

    def test_auth_token_overrides_legacy(self):
        os.environ[plugin.AUTH_TOKEN_ENV] = SYNTHETIC_TOKEN
        os.environ[plugin.TOKEN_ENV] = "cfut_test_legacy_token"
        self.assertEqual(plugin.api_token(), SYNTHETIC_TOKEN)

    def test_legacy_account_used_when_auth_missing(self):
        os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
        os.environ[plugin.ACCOUNT_ENV] = "legacy-acct-456"
        self.assertEqual(plugin.account_id(), "legacy-acct-456")

    def test_legacy_token_used_when_auth_missing(self):
        os.environ.pop(plugin.AUTH_TOKEN_ENV, None)
        os.environ[plugin.TOKEN_ENV] = "cfut_test_legacy_token"
        self.assertEqual(plugin.api_token(), "cfut_test_legacy_token")

    def test_whitespace_only_auth_account_treated_missing(self):
        os.environ[plugin.AUTH_ACCOUNT_ENV] = "   \t  "
        os.environ[plugin.ACCOUNT_ENV] = "legacy-acct-456"
        self.assertEqual(plugin.account_id(), "legacy-acct-456")

    def test_whitespace_only_auth_token_treated_missing(self):
        os.environ[plugin.AUTH_TOKEN_ENV] = " \n "
        os.environ[plugin.TOKEN_ENV] = "cfut_test_legacy_token"
        self.assertEqual(plugin.api_token(), "cfut_test_legacy_token")

    def test_all_whitespace_account_returns_none(self):
        os.environ[plugin.AUTH_ACCOUNT_ENV] = "   "
        os.environ[plugin.ACCOUNT_ENV] = "\t "
        self.assertIsNone(plugin.account_id())

    def test_all_whitespace_token_returns_none(self):
        os.environ[plugin.AUTH_TOKEN_ENV] = " "
        os.environ[plugin.TOKEN_ENV] = "  "
        self.assertIsNone(plugin.api_token())

    def test_empty_string_returns_none(self):
        os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
        os.environ[plugin.ACCOUNT_ENV] = ""
        self.assertIsNone(plugin.account_id())

    def test_missing_variables_return_none(self):
        os.environ.pop(plugin.AUTH_ACCOUNT_ENV, None)
        os.environ.pop(plugin.ACCOUNT_ENV, None)
        os.environ.pop(plugin.AUTH_TOKEN_ENV, None)
        os.environ.pop(plugin.TOKEN_ENV, None)
        self.assertIsNone(plugin.account_id())
        self.assertIsNone(plugin.api_token())

    def test_env_constant_names_are_exact(self):
        self.assertEqual(plugin.AUTH_ACCOUNT_ENV, "AUTH_CLOUDFLARE_ACCOUNT_ID")
        self.assertEqual(plugin.AUTH_TOKEN_ENV, "AUTH_CLOUDFLARE_API_TOKEN")
        self.assertEqual(plugin.ACCOUNT_ENV, "CLOUDFLARE_ACCOUNT_ID")
        self.assertEqual(plugin.TOKEN_ENV, "CLOUDFLARE_API_TOKEN")

    def test_profile_env_vars_list_both_legacy_and_canonical(self):
        profile = plugin.cloudflare
        self.assertIn(plugin.TOKEN_ENV, profile.env_vars)
        self.assertIn(plugin.ACCOUNT_ENV, profile.env_vars)


if __name__ == "__main__":
    unittest.main(verbosity=2)
