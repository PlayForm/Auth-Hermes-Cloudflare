"""Pytest fixtures for the auth-hermes-cloudflare test suite.

Thin wrapper over ``helpers`` so pytest and ``unittest`` discovery share one
loader implementation. The test modules are plain ``unittest.TestCase``
classes, so the same files run under either runner.
"""

from __future__ import annotations

import pytest

from helpers import FIXTURES_DIR, PLUGIN_ROOT, load_plugin


@pytest.fixture(scope="session")
def plugin():
    """The loaded auth-hermes-cloudflare plugin module."""
    return load_plugin()


@pytest.fixture()
def fixtures_dir():
    """Absolute path of the plugin's ``fixtures/`` directory."""
    return FIXTURES_DIR


@pytest.fixture()
def plugin_root():
    """Absolute path of the plugin directory."""
    return PLUGIN_ROOT
