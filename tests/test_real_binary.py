"""Real auth-cloudflare binary integration tests (task sa-3, machine-dependent).

These tests exercise the REAL ``locate_auth_cloudflare_binary`` locator and
the installed ``auth-cloudflare`` executable - the plugin's binary branch.
Every test skips gracefully when the binary is absent, so CI on machines
without the binary stays green.

Offline contract (verified against the installed binary):

- ``doctor --format json`` with synthetic credentials performs NO network
  I/O: it reports the configured state of account id / api token / endpoint
  and exits 0; with the account id missing it exits 2 while still emitting
  the same diagnostic JSON (which the plugin's ``_run_binary_json`` relay
  surfaces instead of a bare "exited with code 2" error).
- ``catalog get --format json`` for a synthetic 32-hex account has no
  account-scoped cache and falls back to the built-in model list
  (``source: fallback``, ``cache_status: none``) - never a live fetch.

All credentials here are synthetic (``cfut_test_...``); no real .env file is
ever read and no real token value can appear in any assertion output.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from pathlib import Path

from helpers import load_plugin

plugin = load_plugin()

SYNTHETIC_ACCOUNT = "a" * 32  # valid 32-hex shape; no such real account exists
SYNTHETIC_TOKEN = "cfut_test_real_binary_0000"  # never a real credential
_SEMVER = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-.][0-9A-Za-z.]+)?")


class RealBinaryTest(unittest.TestCase):
    """Integration tests against the REAL installed auth-cloudflare binary."""

    def setUp(self):
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)

    def _locate_or_skip(self) -> str:
        bin_path = plugin.locate_auth_cloudflare_binary()
        if bin_path is None:
            self.skipTest("auth-cloudflare binary not installed on this machine")
        return bin_path

    def _set_synthetic_creds(self) -> None:
            # The plain CLOUDFLARE_* names are canonical; setting them to
            # synthetic values guarantees no real credential can leak through
            # the binary's resolution either.
            os.environ[plugin.ACCOUNT_ENV] = SYNTHETIC_ACCOUNT
            os.environ[plugin.TOKEN_ENV] = SYNTHETIC_TOKEN

    def test_real_locator_finds_installed_binary(self):
        bin_path = plugin.locate_auth_cloudflare_binary()
        if bin_path is None:
            self.skipTest("auth-cloudflare binary not installed on this machine")
        path = Path(bin_path)
        self.assertTrue(path.is_file(), f"locator returned non-file {bin_path!r}")
        self.assertTrue(os.access(path, os.X_OK), f"{bin_path!r} is not executable")
        ok, detail = plugin.check_binary_compatibility(bin_path)
        self.assertTrue(ok, f"installed binary failed the version handshake: {detail}")
        # The compatibility detail carries name + protocol evidence.
        self.assertIn(plugin.BINARY_NAME, detail)
        self.assertIn("protocol", detail)

    def test_real_binary_version_handshake(self):
        bin_path = self._locate_or_skip()
        ok, detail = plugin.check_binary_compatibility(bin_path)
        self.assertTrue(ok, detail)
        proc = subprocess.run(
            [bin_path, "version", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        self.assertEqual(proc.returncode, 0)
        info = json.loads(proc.stdout)
        self.assertEqual(info.get("name"), plugin.BINARY_NAME)
        package_version = info.get("package_version")
        self.assertIsInstance(package_version, str)
        self.assertIsNotNone(
            _SEMVER.fullmatch(package_version.strip()),
            f"package_version {package_version!r} is not semver",
        )
        protocol_version = info.get("protocol_version")
        self.assertIsInstance(protocol_version, int)
        self.assertGreaterEqual(protocol_version, 1)

    def test_real_binary_doctor_binary_source(self):
        self._locate_or_skip()
        self._set_synthetic_creds()
        result = plugin.cloudflare_doctor()
        self.assertEqual(result.get("source"), "binary")
        for key in ("account_id", "api_token", "endpoint"):
            self.assertIn(key, result)
        self.assertEqual(result["account_id"].get("configured"), True)
        self.assertEqual(result["api_token"].get("configured"), True)
        self.assertEqual(result.get("status"), "ok")
        rendered = json.dumps(result)
        self.assertNotIn(SYNTHETIC_TOKEN, rendered)
        self.assertNotIn(SYNTHETIC_ACCOUNT, rendered)

    def test_real_binary_missing_creds_exits_2_relayed(self):
        self._locate_or_skip()
        self._set_synthetic_creds()
        # Only the token stays configured; the account env is cleared.
        os.environ.pop(plugin.ACCOUNT_ENV, None)
        result = plugin.cloudflare_doctor()
        self.assertEqual(result.get("source"), "binary")
        self.assertEqual(result.get("exit_code"), 2)
        self.assertEqual(result.get("status"), "error")
        # The _run_binary_json relay fix: the binary's diagnostic JSON is
        # surfaced (account_id/api_token/endpoint present, configured:false
        # visible) - NOT a bare "exited with code 2" error object.
        self.assertIn("account_id", result)
        self.assertEqual(result["account_id"].get("configured"), False)
        self.assertIn("api_token", result)
        self.assertIn("endpoint", result)
        self.assertNotIn("error", result)
        rendered = json.dumps(result)
        self.assertNotIn(SYNTHETIC_TOKEN, rendered)

    def test_real_binary_catalog_get_offline(self):
        bin_path = self._locate_or_skip()
        self._set_synthetic_creds()
        # catalog get for a synthetic account reads the account-scoped cache
        # (which does not exist) and falls back to the built-in model list -
        # verified offline-safe (source "fallback", no live fetch). Runs
        # through the plugin's own _run_binary_json so the timeout bound
        # applies and a hang is impossible.
        data, error, rc = plugin._run_binary_json(
            bin_path, ["catalog", "get", "--format", "json"], timeout=15.0
        )
        self.assertIsNone(error, f"catalog get failed: {error}")
        self.assertEqual(rc, 0)
        self.assertIsInstance(data, dict)
        models = data.get("models")
        self.assertIsInstance(models, list)
        self.assertGreaterEqual(len(models), 1)
        # Offline proof: the binary itself reports the fallback source with no
        # cache for the synthetic account - never a live fetch.
        self.assertEqual(data.get("source"), "fallback")
        rendered = json.dumps(data)
        self.assertNotIn(SYNTHETIC_ACCOUNT, rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
