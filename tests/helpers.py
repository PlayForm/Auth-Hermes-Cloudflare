"""Shared helpers for the auth-hermes-cloudflare test suite.

The plugin's ``__init__.py`` lives in a directory whose name contains
hyphens, so it cannot be imported as a normal package. These helpers load it
directly with ``importlib`` (the same approach the Aphrodite-Hermes plugin
tests use) and make Hermes' ``providers`` / ``hermes_cli`` packages
importable when the suite runs outside the Hermes runtime.

Importing the plugin performs NO network I/O: its only import side effect is
``register_provider(...)`` (pure Python). Binary download is an explicit,
separate step (``download.sh``) and is never triggered by import - see
``test_binary_discovery.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PLUGIN_ROOT / "fixtures"

_PLUGIN_MODULE_NAME = "__auth_hermes_cloudflare_under_test"

_HERMES_SRC_CANDIDATES = (
    Path.home() / ".hermes" / "hermes-agent",
    Path.home() / ".hermes" / "agent",
)


def ensure_hermes_importable() -> None:
    """Make Hermes' ``providers``/``hermes_cli`` packages importable.

    No-op when the suite runs inside the Hermes runtime (the packages are
    already importable). Otherwise prepends the first discovered Hermes agent
    source tree to ``sys.path`` - ``HERMES_AGENT_SRC`` env var wins, then
    well-known locations. Fails loudly rather than stubbing Hermes internals,
    so the tests always exercise the real plugin against the real base
    classes.
    """
    try:
        import providers  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    env_src = os.getenv("HERMES_AGENT_SRC")
    candidates = [Path(env_src)] if env_src else []
    candidates.extend(_HERMES_SRC_CANDIDATES)
    for cand in candidates:
        if (cand / "providers" / "__init__.py").is_file():
            sys.path.insert(0, str(cand))
            return
    raise RuntimeError(
        "Could not locate the Hermes agent source tree needed to import the "
        "plugin's `providers`/`hermes_cli` packages. Set HERMES_AGENT_SRC or "
        "run with a Hermes installation present at ~/.hermes/hermes-agent."
    )


def load_plugin(*, fresh: bool = False):
    """Load the plugin's ``__init__.py`` as a module and return it.

    Cached across test modules (one profile registration per run); pass
    ``fresh=True`` to force a reload (used by the import-safety test).
    """
    ensure_hermes_importable()
    if not fresh and _PLUGIN_MODULE_NAME in sys.modules:
        return sys.modules[_PLUGIN_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(
        _PLUGIN_MODULE_NAME, PLUGIN_ROOT / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[_PLUGIN_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def fixture_path(name: str) -> Path:
    """Absolute path of a fixture file under the plugin's ``fixtures/`` dir."""
    return FIXTURES_DIR / name
