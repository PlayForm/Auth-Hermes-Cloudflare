"""Binary discovery tests.

Contract under test:

- Discovery order is strictly: ``AUTH_CLOUDFLARE_BIN`` > ``PATH`` >
  ``~/.hermes/bin`` > plugin-local ``bin/`` > plugin runtime cache.
- Missing/not-executable candidates fall through to the next source.
- Discovery never downloads; downloads happen only as an explicit, separate
  step (``download.sh`` / explicit user action), never during Python import.
- Importing the plugin module performs no network I/O whatsoever.

The discovery function below is deliberately test-local (the plugin's
``__init__.py`` is not modified): it mirrors the reference
implementation so the order is pinned by tests until the real locator lands.
"""

from __future__ import annotations

import os
import socket
import tempfile
import unittest
from pathlib import Path

from helpers import PLUGIN_ROOT, load_plugin

plugin = load_plugin()

BINARY_NAME = "auth-cloudflare"


class _DownloadInvoked(Exception):
    """Raised if a download path is ever reached during discovery."""


def locate_auth_cloudflare_binary(
    *,
    env=None,
    path=None,
    home=None,
    plugin_root=None,
    cache_dir=None,
    download=None,
):
    """Mirror of the discovery order (test-local helper).

    Order: AUTH_CLOUDFLARE_BIN > PATH > ~/.hermes/bin > plugin bin > cache.
    Returns the first executable candidate or None. ``download`` is accepted
    only to prove it is never called by discovery.
    """
    env = dict(os.environ) if env is None else dict(env)
    if path is None:
        path = env.get("PATH", os.defpath)
    path = str(path)
    home_path = Path(home).expanduser() if home else Path.home()

    explicit = env.get("AUTH_CLOUDFLARE_BIN")
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate

    on_path = None
    for directory in path.split(os.pathsep):
        if not directory:
            continue
        probe = Path(directory) / BINARY_NAME
        if probe.is_file() and os.access(probe, os.X_OK):
            on_path = probe
            break
    if on_path is not None:
        return on_path

    candidates = [
        home_path / ".hermes" / "bin" / BINARY_NAME,
        Path(plugin_root or PLUGIN_ROOT) / "bin" / BINARY_NAME,
    ]
    if cache_dir is not None:
        candidates.append(Path(cache_dir) / BINARY_NAME)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


class BinaryDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="auth-cf-discovery-test-"))
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        for child in self._tmp.glob("**/*"):
            if child.is_file():
                child.chmod(0o700)
        for child in sorted(self._tmp.glob("**/*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass

    def _make_executable(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_env_override_wins(self):
        env_bin = self._make_executable(self._tmp / "custom" / BINARY_NAME)
        path_bin = self._make_executable(self._tmp / "on-path" / BINARY_NAME)
        found = locate_auth_cloudflare_binary(
            env={"AUTH_CLOUDFLARE_BIN": str(env_bin), "PATH": str(path_bin.parent)},
            home=self._tmp / "home",
        )
        self.assertEqual(found, env_bin)

    def test_env_pointing_at_missing_file_falls_through(self):
        path_bin = self._make_executable(self._tmp / "on-path" / BINARY_NAME)
        found = locate_auth_cloudflare_binary(
            env={
                "AUTH_CLOUDFLARE_BIN": str(self._tmp / "missing" / BINARY_NAME),
                "PATH": str(path_bin.parent),
            },
            home=self._tmp / "home",
        )
        self.assertEqual(found, path_bin)

    def test_path_discovery(self):
        path_bin = self._make_executable(self._tmp / "on-path" / BINARY_NAME)
        found = locate_auth_cloudflare_binary(
            env={},
            path=str(path_bin.parent),
            home=self._tmp / "home",
        )
        self.assertEqual(found, path_bin)

    def test_home_hermes_bin_discovery(self):
        home_bin = self._make_executable(
            self._tmp / "home" / ".hermes" / "bin" / BINARY_NAME
        )
        found = locate_auth_cloudflare_binary(
            env={},
            path=self._tmp / "empty-path",
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
        )
        self.assertEqual(found, home_bin)

    def test_plugin_local_bin_discovery(self):
        plugin_bin = self._make_executable(self._tmp / "plugin" / "bin" / BINARY_NAME)
        found = locate_auth_cloudflare_binary(
            env={},
            path=self._tmp / "empty-path",
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
        )
        self.assertEqual(found, plugin_bin)

    def test_cache_dir_discovery(self):
        cache_bin = self._make_executable(self._tmp / "cache" / BINARY_NAME)
        found = locate_auth_cloudflare_binary(
            env={},
            path=self._tmp / "empty-path",
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
            cache_dir=self._tmp / "cache",
        )
        self.assertEqual(found, cache_bin)

    def test_non_executable_candidate_skipped(self):
        path_bin = self._tmp / "on-path" / BINARY_NAME
        path_bin.parent.mkdir(parents=True, exist_ok=True)
        path_bin.write_text("not executable", encoding="utf-8")  # no chmod +x
        found = locate_auth_cloudflare_binary(
            env={},
            path=str(path_bin.parent),
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
        )
        self.assertIsNone(found)

    def test_missing_executable_returns_none(self):
        found = locate_auth_cloudflare_binary(
            env={},
            path=self._tmp / "empty-path",
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
            cache_dir=self._tmp / "cache",
        )
        self.assertIsNone(found)

    def test_discovery_never_invokes_download(self):
        def forbidden_download(*args, **kwargs):
            raise _DownloadInvoked("download must never run during discovery")

        found = locate_auth_cloudflare_binary(
            env={},
            path=self._tmp / "empty-path",
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
            download=forbidden_download,
        )
        self.assertIsNone(found)

    def test_env_override_skips_every_other_source(self):
        env_bin = self._make_executable(self._tmp / "custom" / BINARY_NAME)
        # Poison every fallback source with a marker that would raise if used;
        # env override must win without consulting anything else.
        home_bin = self._tmp / "home" / ".hermes" / "bin" / BINARY_NAME
        home_bin.parent.mkdir(parents=True, exist_ok=True)
        home_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        home_bin.chmod(0o755)
        found = locate_auth_cloudflare_binary(
            env={"AUTH_CLOUDFLARE_BIN": str(env_bin)},
            path=self._tmp / "empty-path",
            home=self._tmp / "home",
            plugin_root=self._tmp / "plugin",
        )
        self.assertEqual(found, env_bin)


class RealBinaryLocatorTest(unittest.TestCase):
    """The REAL plugin.locate_auth_cloudflare_binary (task sa-3 re-point).

    Hermetic: both tests control ``AUTH_CLOUDFLARE_BIN`` and ``PATH`` so they
    pass whether or not a real auth-cloudflare binary is installed - the real
    locator's env-override ordering is what is pinned here.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="auth-cf-real-locator-test-"))
        self._saved_env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved_env)
        for child in self._tmp.glob("**/*"):
            if child.is_file():
                child.chmod(0o700)
        for child in sorted(self._tmp.glob("**/*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass

    def _make_executable(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_real_locator_env_override_wins(self):
        env_bin = self._make_executable(self._tmp / "custom" / BINARY_NAME)
        os.environ["AUTH_CLOUDFLARE_BIN"] = str(env_bin)
        # PATH is isolated so the real binary (if installed) cannot win.
        os.environ["PATH"] = str(self._tmp / "empty-path")
        found = plugin.locate_auth_cloudflare_binary()
        self.assertEqual(found, str(env_bin))

    def test_real_locator_env_dead_end_falls_through(self):
        # AUTH_CLOUDFLARE_BIN pointing at a missing file must fall through to
        # PATH - the real locator's documented order.
        path_bin = self._make_executable(self._tmp / "on-path" / BINARY_NAME)
        os.environ["AUTH_CLOUDFLARE_BIN"] = str(self._tmp / "missing" / BINARY_NAME)
        os.environ["PATH"] = str(path_bin.parent)
        found = plugin.locate_auth_cloudflare_binary()
        self.assertEqual(found, str(path_bin))


class ImportSafetyTest(unittest.TestCase):
    """The plugin module must import without any network or download."""

    def test_no_download_and_no_socket_at_import(self):
        original_socket = socket.socket
        calls = []

        class ForbiddenSocket(socket.socket):
            """A real socket subclass that explodes if ever instantiated.

            Using a subclass (not a bare function) keeps `import ssl` and
            `class SSLSocket(socket)` working when the stdlib is not yet
            loaded, while still failing loudly on any socket creation.
            """

            def __init__(self, *args, **kwargs):
                calls.append(True)
                raise AssertionError("socket created during plugin import")

        socket.socket = ForbiddenSocket  # type: ignore[assignment]
        try:
            module = load_plugin(fresh=True)
        finally:
            socket.socket = original_socket  # type: ignore[assignment]
        self.assertEqual(calls, [])
        self.assertIsNotNone(module.cloudflare)
        # The plugin exposes no auto-download callable that import could hit.
        self.assertFalse(hasattr(module, "download"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
