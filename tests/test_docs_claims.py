"""Docs-claims tests (naming + executable-based behavior).

Asserts the shipped docs - the plugin ``README.md``, ``plugin.yaml``, and the
repo root ``README.md`` - no longer claim a standalone ``Cloudflare AI``
provider, never claim ``all Cloudflare AI models``, and that the plugin README
no longer describes a dylib-based flow (the shipped artifact is the
``auth-cloudflare`` executable installed by ``download.sh``).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
ROOT_README = PLUGIN_ROOT.parent.parent / "README.md"

# The display name is allowed verbatim. Note it does not contain the literal
# substring "Cloudflare AI" ("Auth Cloudflare Workers AI" has "Workers " in
# between). "Cloudflare AI Gateway" is a distinct Cloudflare product named in
# the root README's out-of-scope list, not a claim about this provider, so it
# is stripped before scanning for the forbidden shorthand.
_ALLOWED_PHRASES = (
    "Auth Cloudflare Workers AI",
    "Cloudflare AI Gateway",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _standalone_claims(text: str) -> list[str]:
    """Every literal ``Cloudflare AI`` substring outside the allowed phrases."""
    sanitized = text
    for phrase in _ALLOWED_PHRASES:
        sanitized = sanitized.replace(phrase, "")
    return re.findall(r"Cloudflare AI", sanitized)


class DocsClaimsTest(unittest.TestCase):
    """The shipped docs must not claim a standalone ``Cloudflare AI`` provider."""

    @classmethod
    def setUpClass(cls):
        cls.docs = {
            "plugin README": PLUGIN_ROOT / "README.md",
            "plugin.yaml": PLUGIN_ROOT / "plugin.yaml",
            "root README": ROOT_README,
        }

    def test_docs_exist(self):
        for label, path in self.docs.items():
            self.assertTrue(path.is_file(), f"{label} missing: {path}")

    def test_no_standalone_cloudflare_ai_claim(self):
        for label, path in self.docs.items():
            claims = _standalone_claims(_read(path))
            self.assertEqual(
                claims,
                [],
                f"{label} contains a standalone 'Cloudflare AI' claim: {claims}",
            )

    def test_no_all_cloudflare_ai_models_claim(self):
        for label, path in self.docs.items():
            self.assertNotIn(
                "all Cloudflare AI models",
                _read(path),
                f"{label} claims 'all Cloudflare AI models'",
            )

    def test_plugin_readme_no_dylib_flow(self):
        text = _read(PLUGIN_ROOT / "README.md")
        self.assertNotIn(
            "dylib",
            text,
            "plugin README still mentions 'dylib' - the shipped artifact is the "
            "auth-cloudflare executable, not a dylib",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
