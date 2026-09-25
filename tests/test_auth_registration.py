"""Credential registration (``hermes cloudflare auth``) tests.

Contract under test:

- ``ProviderProfile.env_vars`` must NOT contain the account-id env var: stock
  ``hermes_cli.auth_plugin_providers._api_key_env_fields`` treats every
  non-URL env var as an api-key credential, and an account id seeded into
  the credential pool is tried as an API key after the real token is
  rate-limited -> the exact 401 ``Authentication error`` delegated agents
  saw.
- ``api_token()`` falls back to ``~/.hermes/.env`` and then to the
  credential pool, so subagent processes with a fresh environment resolve
  the key from disk.
- ``cloudflare_auth()`` registers the token into the credential pool under
  the canonical provider name and the ``cloudflare`` alias, prunes junk
  account-id entries, and clears exhaustion.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from helpers import load_plugin

plugin = load_plugin()

SYNTHETIC_TOKEN = "cfut_test_auth_registration"  # never a real credential
SYNTHETIC_ACCOUNT = "a" * 32


class FakePool:
    """Minimal in-memory stand-in for agent.credential_pool.CredentialPool."""

    def __init__(self, entries=None):
        self._entries = list(entries or [])
        self.added = []
        self.removed = []
        self.resets = 0

    def entries(self):
        return list(self._entries)

    def resolve_target(self, target):
        for idx, entry in enumerate(self._entries, start=1):
            if entry.id == target:
                return idx, entry, None
        return None, None, "not found"

    def remove_index(self, index):
        if 1 <= index <= len(self._entries):
            self.removed.append(self._entries.pop(index - 1))

    def add_entry(self, entry):
        self.added.append(entry)
        self._entries.append(entry)
        return entry

    def reset_statuses(self):
        self.resets += 1
        return 1


def _pool_entry(label, *, token=None, account=False):
    return SimpleNamespace(
        id="x" + label[:4].lower() + "ab",
        label=label,
        runtime_api_key=token or (SYNTHETIC_ACCOUNT if account else ""),
    )


def _install_fake_credential_pool(pools):
    """Inject a hermetic ``agent.credential_pool`` into ``sys.modules``.

    The real module drags in ``ruamel.yaml`` through ``hermes_yaml``, which
    the ephemeral test env does not provide. The plugin imports the pool
    names at call time, so a stub module with the same names is sufficient.
    Returns the previous ``sys.modules`` entries for restoration.
    """
    import sys
    from types import ModuleType

    saved_agent = sys.modules.get("agent")
    saved_pool = sys.modules.get("agent.credential_pool")

    agent = ModuleType("agent")
    fake_pool = ModuleType("agent.credential_pool")
    fake_pool.AUTH_TYPE_API_KEY = "api_key"
    fake_pool.SOURCE_MANUAL = "manual"
    fake_pool.PooledCredential = SimpleNamespace
    fake_pool.load_pool = lambda provider: pools.get(provider)
    sys.modules["agent"] = agent
    sys.modules["agent.credential_pool"] = fake_pool
    return saved_agent, saved_pool


def _restore_credential_pool(saved_agent, saved_pool):
    import sys

    if saved_agent is not None:
        sys.modules["agent"] = saved_agent
    else:
        sys.modules.pop("agent", None)
    if saved_pool is not None:
        sys.modules["agent.credential_pool"] = saved_pool
    else:
        sys.modules.pop("agent.credential_pool", None)


class EnvVarsShapeTest(unittest.TestCase):
    def test_env_vars_exclude_account_id(self):
        # The account id is an auth PARAMETER, never an api-key credential.
        self.assertNotIn(plugin.ACCOUNT_ENV, plugin.cloudflare.env_vars)
        self.assertNotIn(plugin.ACCOUNT_ENV, plugin.cloudflare.env_vars)

    def test_env_vars_keep_token_and_base_url(self):
        self.assertIn(plugin.TOKEN_ENV, plugin.cloudflare.env_vars)
        self.assertIn(plugin.BASE_URL_ENV, plugin.cloudflare.env_vars)


class ApiTokenFallbackTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["HOME"] = self._tmp.name
        # Ambient real credentials must not leak in: the plain names are the
        # sole credential contract now.
        os.environ.pop(plugin.ACCOUNT_ENV, None)
        os.environ.pop(plugin.TOKEN_ENV, None)
        # Isolate the pool fallback: the real auth.json must not leak in.
        self._saved_pool = plugin._pool_api_token
        plugin._pool_api_token = lambda provider: None

    def tearDown(self):
        plugin._pool_api_token = self._saved_pool
        os.environ.clear()
        os.environ.update(self._saved)
        self._tmp.cleanup()

    def _write_dotenv(self, content):
        env_dir = Path(self._tmp.name) / ".hermes"
        env_dir.mkdir(parents=True, exist_ok=True)
        (env_dir / ".env").write_text(content, encoding="utf-8")

    def test_api_token_dotenv_fallback(self):
        self._write_dotenv(f"{plugin.TOKEN_ENV}={SYNTHETIC_TOKEN}\n")
        self.assertEqual(plugin.api_token(), SYNTHETIC_TOKEN)

    def test_account_id_dotenv_fallback(self):
        self._write_dotenv(f"{plugin.ACCOUNT_ENV}={SYNTHETIC_ACCOUNT}\n")
        self.assertEqual(plugin.account_id(), SYNTHETIC_ACCOUNT)

    def test_missing_dotenv_returns_none(self):
        self.assertIsNone(plugin.api_token())
        self.assertIsNone(plugin.account_id())

    def test_env_wins_over_dotenv(self):
        self._write_dotenv(f"{plugin.TOKEN_ENV}=cfut_dotenv_value\n")
        os.environ[plugin.TOKEN_ENV] = SYNTHETIC_TOKEN
        self.assertEqual(plugin.api_token(), SYNTHETIC_TOKEN)


class ApiTokenPoolFallbackTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["HOME"] = self._tmp.name
        # Ambient real credentials must not leak in: the plain names are the
        # sole credential contract now.
        os.environ.pop(plugin.ACCOUNT_ENV, None)
        os.environ.pop(plugin.TOKEN_ENV, None)
        self._saved_dotenv = plugin._dotenv_value
        plugin._dotenv_value = lambda key: None

    def tearDown(self):
        plugin._dotenv_value = self._saved_dotenv
        os.environ.clear()
        os.environ.update(self._saved)
        self._tmp.cleanup()

    def test_api_token_pool_fallback(self):
        saved_agent, saved_pool = _install_fake_credential_pool(
            {
                "auth-cloudflare-workers-ai": FakePool(
                    [_pool_entry(plugin.TOKEN_ENV, token=SYNTHETIC_TOKEN)]
                )
            }
        )
        try:
            self.assertEqual(plugin.api_token(), SYNTHETIC_TOKEN)
        finally:
            _restore_credential_pool(saved_agent, saved_pool)


class CloudflareAuthCommandTest(unittest.TestCase):
    def setUp(self):
        self._saved = dict(os.environ)
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["HOME"] = self._tmp.name
        # Deterministic credential resolution, no .env writes, no live catalog.
        self._saved_token = plugin.api_token
        self._saved_account = plugin.account_id
        self._saved_dotenv = plugin._dotenv_value
        self._saved_persist = plugin._persist_env_value
        self._saved_validate = plugin.validate_setup
        plugin.api_token = lambda: SYNTHETIC_TOKEN
        plugin.account_id = lambda: SYNTHETIC_ACCOUNT
        plugin._dotenv_value = lambda key: None
        plugin._persist_env_value = lambda key, value: True
        plugin.validate_setup = lambda: {"overall": "ok"}

    def tearDown(self):
        plugin.api_token = self._saved_token
        plugin.account_id = self._saved_account
        plugin._dotenv_value = self._saved_dotenv
        plugin._persist_env_value = self._saved_persist
        plugin.validate_setup = self._saved_validate
        os.environ.clear()
        os.environ.update(self._saved)
        self._tmp.cleanup()

    def test_registers_into_pool_for_canonical_and_alias(self):
        pools = {
            "auth-cloudflare-workers-ai": FakePool(
                [
                    _pool_entry(plugin.TOKEN_ENV, token="cfut_stale"),
                    _pool_entry(plugin.ACCOUNT_ENV, account=True),
                ]
            ),
            "cloudflare": FakePool([_pool_entry(plugin.ACCOUNT_ENV, account=True)]),
        }
        saved_agent, saved_pool = _install_fake_credential_pool(pools)
        try:
            result = plugin.cloudflare_auth()
        finally:
            _restore_credential_pool(saved_agent, saved_pool)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            set(result["providers"]), {"auth-cloudflare-workers-ai", "cloudflare"}
        )
        # Junk account-id entries pruned under both providers.
        self.assertEqual(len(pools["auth-cloudflare-workers-ai"].removed), 2)
        self.assertEqual(len(pools["cloudflare"].removed), 1)
        # One fresh manual entry per provider, carrying the resolved token.
        for provider in ("auth-cloudflare-workers-ai", "cloudflare"):
            added = pools[provider].added
            self.assertEqual(len(added), 1)
            self.assertEqual(added[0].access_token, SYNTHETIC_TOKEN)
            self.assertEqual(added[0].provider, provider)
        # Exhaustion was cleared on both pools.
        self.assertEqual(pools["auth-cloudflare-workers-ai"].resets, 1)
        self.assertEqual(pools["cloudflare"].resets, 1)

    def test_missing_token_errors_without_touching_pool(self):
        touched = []
        saved_agent, saved_pool = _install_fake_credential_pool({})
        # Track which providers load_pool was asked for.
        import sys

        sys.modules["agent.credential_pool"].load_pool = lambda provider: (
            touched.append(provider) or None
        )
        saved_token = plugin.api_token
        plugin.api_token = lambda: None
        try:
            result = plugin.cloudflare_auth()
        finally:
            plugin.api_token = saved_token
            _restore_credential_pool(saved_agent, saved_pool)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(touched, [])

    def test_missing_account_errors_without_touching_pool(self):
        touched = []
        saved_agent, saved_pool = _install_fake_credential_pool({})
        import sys

        sys.modules["agent.credential_pool"].load_pool = lambda provider: (
            touched.append(provider) or None
        )
        saved_account = plugin.account_id
        plugin.account_id = lambda: None
        try:
            result = plugin.cloudflare_auth()
        finally:
            plugin.account_id = saved_account
            _restore_credential_pool(saved_agent, saved_pool)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(touched, [])


class AuthRouterTest(unittest.TestCase):
    def test_router_dispatches_auth(self):
        saved = plugin.cloudflare_auth
        plugin.cloudflare_auth = lambda **kwargs: {"status": "ok", "routed": kwargs}
        try:
            result = plugin.cloudflare_command("auth")
        finally:
            plugin.cloudflare_auth = saved
        self.assertEqual(result["status"], "ok")

    def test_router_unknown_command_unchanged(self):
        result = plugin.cloudflare_command("bogus")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
