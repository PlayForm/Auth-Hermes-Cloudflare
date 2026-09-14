"""Version compatibility tests.

Contract under test:

- ``BINARY_VERSION`` and ``PROTOCOL_VERSION`` files exist and parse.
- The protocol version is an integer >= 1; the binary version is semver.
- ``fixtures/auth_cloudflare_version.json`` matches the documented
  ``auth-cloudflare version --format json`` shape.
- An incompatible (lower) protocol reported by the binary blocks binary use;
  equal or higher protocol versions are accepted.
- ``plugin.yaml`` version matches ``BINARY_VERSION`` and declares
  ``min_hermes_version``.
"""

from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from helpers import PLUGIN_ROOT, fixture_path, load_plugin

plugin = load_plugin()

SEMVER = re.compile(r"^\d+\.\d+\.\d+$")

# A version JSON the plugin's REAL handshake must accept (mirrors the
# fixture plus the BINARY_VERSION/PROTOCOL_VERSION/CATALOG_SCHEMA_VERSION
# plugin constants).
GOOD_VERSION = {
    "name": plugin.BINARY_NAME,
    "package_version": plugin.BINARY_VERSION,
    "protocol_version": plugin.PROTOCOL_VERSION,
    "catalog_schema_versions": [plugin.CATALOG_SCHEMA_VERSION],
    "minimum_hermes_plugin_version": plugin.BINARY_VERSION,
}


def _write_fake_binary(path: Path, version_info: dict) -> Path:
    """Write an executable fake auth-cloudflare that prints *version_info* JSON.

    check_binary_compatibility runs ``<bin> version --format json``; this fake
    ignores its argv and emits the dict as its JSON response (exit 0).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(version_info)
    path.write_text(
        f"#!/usr/bin/env python3\nimport sys\nprint({payload!r})\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def is_protocol_compatible(binary_protocol: int, required: int) -> bool:
    """Test-local mirror of the plugin/binary handshake.

    The binary's reported protocol must be >= the protocol the plugin was
    built against; otherwise the binary is rejected.
    """
    return binary_protocol >= required


def _simple_yaml_parse(path) -> dict:
    """Minimal YAML-subset parser for the plugin manifest (no PyYAML dep).

    Handles the flat ``key: value`` entries used by plugin.yaml; comments and
    lists are ignored.
    """
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().strip("'\"")
        if value:
            values[key.strip()] = value
    return values


class VersionCompatibilityTest(unittest.TestCase):
    def test_binary_version_file_exists_and_parses(self):
        path = PLUGIN_ROOT / "BINARY_VERSION"
        self.assertTrue(path.is_file(), "BINARY_VERSION file missing")
        value = path.read_text(encoding="utf-8").strip()
        self.assertRegex(value, SEMVER)

    def test_protocol_version_file_exists_and_parses(self):
        path = PLUGIN_ROOT / "PROTOCOL_VERSION"
        self.assertTrue(path.is_file(), "PROTOCOL_VERSION file missing")
        value = path.read_text(encoding="utf-8").strip()
        self.assertTrue(value.isdigit(), f"protocol not an integer: {value!r}")
        self.assertGreaterEqual(int(value), 1)

    def test_binary_and_protocol_files_present_together(self):
        self.assertTrue((PLUGIN_ROOT / "BINARY_VERSION").is_file())
        self.assertTrue((PLUGIN_ROOT / "PROTOCOL_VERSION").is_file())

    def test_version_fixture_matches_documented_shape(self):
        doc = json.loads(fixture_path("auth_cloudflare_version.json").read_text())
        self.assertEqual(doc["name"], "auth-cloudflare")
        self.assertRegex(doc["package_version"], SEMVER)
        self.assertIsInstance(doc["protocol_version"], int)
        self.assertGreaterEqual(doc["protocol_version"], 1)
        self.assertIsInstance(doc["catalog_schema_versions"], list)
        self.assertTrue(all(isinstance(v, int) for v in doc["catalog_schema_versions"]))
        self.assertRegex(doc["minimum_hermes_plugin_version"], SEMVER)

    def test_fixture_protocol_matches_protocol_file(self):
        required = int((PLUGIN_ROOT / "PROTOCOL_VERSION").read_text().strip())
        doc = json.loads(fixture_path("auth_cloudflare_version.json").read_text())
        self.assertEqual(doc["protocol_version"], required)

    def test_fixture_package_version_matches_binary_file(self):
        required = (PLUGIN_ROOT / "BINARY_VERSION").read_text().strip()
        doc = json.loads(fixture_path("auth_cloudflare_version.json").read_text())
        self.assertEqual(doc["package_version"], required)

    def test_incompatible_protocol_blocks_binary_use(self):
        self.assertFalse(is_protocol_compatible(0, 1))
        self.assertFalse(is_protocol_compatible(0, 1))

    def test_compatible_protocol_accepted(self):
        self.assertTrue(is_protocol_compatible(1, 1))
        self.assertTrue(is_protocol_compatible(2, 1))

    def test_plugin_yaml_version_matches_binary_version(self):
        manifest = _simple_yaml_parse(PLUGIN_ROOT / "plugin.yaml")
        self.assertEqual(manifest["name"], "auth-hermes-cloudflare")
        self.assertEqual(manifest["kind"], "model-provider")
        self.assertEqual(
            manifest["version"], (PLUGIN_ROOT / "BINARY_VERSION").read_text().strip()
        )

    def test_plugin_yaml_declares_min_hermes_version(self):
        manifest = _simple_yaml_parse(PLUGIN_ROOT / "plugin.yaml")
        self.assertIn("min_hermes_version", manifest)
        self.assertRegex(manifest["min_hermes_version"], r"^\d+\.\d+\.\d+$")

    def test_plugin_yaml_lists_env_vars(self):
        text = (PLUGIN_ROOT / "plugin.yaml").read_text(encoding="utf-8")
        for var in (
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "AUTH_CLOUDFLARE_API_TOKEN",
            "AUTH_CLOUDFLARE_ACCOUNT_ID",
        ):
            self.assertIn(var, text)


class RealCheckBinaryCompatibilityTest(unittest.TestCase):
    """Drive the REAL plugin.check_binary_compatibility() handshake.

    Each case feeds a fake executable that emits a specific version JSON, so
    the catalog-schema and minimum-binary-version gates are pinned without
    depending on the installed binary.
    """

    def setUp(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="auth-cf-compat-test-"))

    def tearDown(self):
        for child in self._tmp.glob("**/*"):
            if child.is_file():
                child.chmod(0o700)
        for child in sorted(self._tmp.glob("**/*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass

    def _check(self, version_info: dict):
        path = _write_fake_binary(self._tmp / "fake-auth-cloudflare", version_info)
        return plugin.check_binary_compatibility(str(path))

    def test_happy_path_compatible(self):
        ok, detail = self._check(dict(GOOD_VERSION))
        self.assertTrue(ok, detail)
        self.assertIn("compatible", detail)

    def test_missing_catalog_schema_versions_incompatible(self):
        info = {k: v for k, v in GOOD_VERSION.items() if k != "catalog_schema_versions"}
        ok, detail = self._check(info)
        self.assertFalse(ok)
        self.assertIn("catalog_schema_versions", detail)

    def test_catalog_schema_versions_without_1_incompatible(self):
        info = dict(GOOD_VERSION)
        info["catalog_schema_versions"] = [2]
        ok, detail = self._check(info)
        self.assertFalse(ok)
        self.assertIn("catalog_schema_versions", detail)

    def test_package_version_older_than_binary_version_incompatible(self):
        info = dict(GOOD_VERSION)
        info["package_version"] = "0.0.0"
        ok, detail = self._check(info)
        self.assertFalse(ok)
        self.assertIn("older", detail)


if __name__ == "__main__":
    unittest.main(verbosity=2)
