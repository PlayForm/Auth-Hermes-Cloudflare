"""Token redaction tests.

Contract under test: the API token never appears in ``repr``/``str`` of the
profile or in the catalog-fetch result/log path. ``fetch_models`` is
binary-first and falls back to the static ``FALLBACK_MODELS`` list - it never
sends the token over HTTP itself (the ``auth-cloudflare`` executable owns the
Bearer header). The token's only intentional exposure is the
``Authorization`` header the binary sends to Cloudflare.

All tokens used here are synthetic (``cfut_test_...``) - never real
credentials.
"""

from __future__ import annotations

import logging
import os
import unittest
import urllib.error
import urllib.request
from unittest import mock

from helpers import load_plugin

plugin = load_plugin()

ACCOUNT = "testacct0001"
TOKEN = "cfut_test_" + "x" * 32  # synthetic, never a real credential


class TokenRedactionTest(unittest.TestCase):
    def setUp(self):
        self._saved_env = dict(os.environ)
        os.environ[plugin.ACCOUNT_ENV] = ACCOUNT
        os.environ[plugin.TOKEN_ENV] = TOKEN
        # Force the static-fallback path (the binary is installed on dev
        # machines and is binary-first; delegation is covered by
        # test_binary_discovery.py).
        self._locator = mock.patch.object(
            plugin, "locate_auth_cloudflare_binary", return_value=None
        )
        self._locator.start()

    def tearDown(self):
        self._locator.stop()
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _fresh_profile(self):
        # A standalone instance with a unique name so nothing from the
        # registered module-level profile interferes with the assertions.
        return plugin.CloudflareProfile(
            name="auth-cloudflare-workers-ai-redaction-test",
            display_name="Auth Cloudflare Workers AI",
            env_vars=(plugin.TOKEN_ENV, plugin.ACCOUNT_ENV),
        )

    def test_repr_hides_token(self):
        profile = self._fresh_profile()
        self.assertNotIn(TOKEN, repr(profile))

    def test_str_hides_token(self):
        profile = self._fresh_profile()
        self.assertNotIn(TOKEN, str(profile))

    def test_module_profile_repr_hides_token(self):
        # The registered module-level instance must also never expose it.
        self.assertNotIn(TOKEN, repr(plugin.cloudflare))
        self.assertNotIn(TOKEN, str(plugin.cloudflare))

    def test_no_profile_attribute_holds_token(self):
        profile = self._fresh_profile()
        for field_name, value in vars(profile).items():
            if isinstance(value, str):
                self.assertNotIn(TOKEN, value, f"field {field_name} leaked the token")

    def test_token_env_var_is_not_in_profile_repr(self):
        # The canonical env var NAME may appear (setup guidance) but never the
        # value. Assert the literal value is absent from every printable form.
        profile = self._fresh_profile()
        rendered = repr(profile) + str(profile)
        self.assertNotIn(TOKEN, rendered)

    def test_fetch_fallback_path_logs_no_token(self):
        # The static-fallback catalog path never touches the network, so the
        # synthetic token must not reach any formatted log record.
        records = []
        handler = logging.Handler()
        handler.emit = lambda record: records.append(record)
        logger = logging.getLogger("providers.base")
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            result = plugin.cloudflare.fetch_models()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)
        self.assertEqual(result, list(plugin.FALLBACK_MODELS))
        for record in records:
            formatted = record.getMessage()
            self.assertNotIn(TOKEN, formatted)
            self.assertNotIn("Bearer " + TOKEN, formatted)

    def test_http_error_repr_does_not_contain_token(self):
        # The only place the token travels is the Authorization header, so an
        # error raised from a request object must be token-free.
        req = urllib.request.Request(plugin.catalog_url())
        req.add_header("Authorization", f"Bearer {TOKEN}")
        exc = urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, None)
        self.assertNotIn(TOKEN, repr(exc))
        self.assertNotIn(TOKEN, str(exc))

    def test_request_repr_does_not_contain_token(self):
        req = urllib.request.Request(plugin.catalog_url())
        req.add_header("Authorization", f"Bearer {TOKEN}")
        self.assertNotIn(TOKEN, repr(req))

    def test_api_token_accessor_is_the_intentional_exposure(self):
        # The accessor must return the configured value - it feeds the Bearer
        # header the binary sends. This is the single sanctioned path;
        # everything printable around the profile stays clean.
        self.assertEqual(plugin.api_token(), TOKEN)

    def test_fetch_success_keeps_token_out_of_result(self):
        # Even on success the returned model list is token-free.
        result = plugin.cloudflare.fetch_models()
        self.assertEqual(result, list(plugin.FALLBACK_MODELS))
        for model_id in result:
            self.assertNotIn(TOKEN, model_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
