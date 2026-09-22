"""auth-hermes-cloudflare - Cloudflare Workers AI model-provider plugin for Hermes Agent.

Registers the ``auth-cloudflare-workers-ai`` provider against Cloudflare's
OpenAI-compatible Workers AI surface:

- Inference:  ``POST /client/v4/accounts/<ACCOUNT_ID>/ai/v1/chat/completions``
- Catalog:    ``GET  /client/v4/accounts/<ACCOUNT_ID>/ai/models/search``
              (OpenRouter-compatible format; NOT OpenAI's ``/models``)
- Verify:     ``GET  /client/v4/user/tokens/verify``

The account ID is injected from the environment (``CLOUDFLARE_ACCOUNT_ID`` /
``AUTH_CLOUDFLARE_ACCOUNT_ID``); the API token (``CLOUDFLARE_API_TOKEN`` /
``AUTH_CLOUDFLARE_API_TOKEN``) is a secret and is only ever sent as a Bearer
header - never logged, never echoed.

The base URL is DERIVED from the account ID. The profile declares
``CLOUDFLARE_BASE_URL`` (a ``*_BASE_URL`` env var) in ``env_vars``, so stock
Hermes ``hermes_cli/auth._register_plugin_provider`` maps it to the
``ProviderConfig.base_url_env_var`` slot; ``hermes cloudflare setup`` writes
the derived URL into ``~/.hermes/.env`` under that name, and the stock wizard
(``_model_flow_api_key_provider``) reads it back as the pre-filled Base URL
default - the user never types a Base URL. A legacy ``fixed_base_url`` flag
is set post-construction only on cores that still declare it (a ``hermes
update`` wipes patched cores, so the plugin must import cleanly on stock
core).
This provider is registered as ``auth-cloudflare-workers-ai`` with the
display name **Auth Cloudflare Workers AI**. Live catalog discovery is owned
by the ``auth-cloudflare`` executable through the JSON CLI protocol
(``catalog get --format json``): when the binary is present and compatible it
is authoritative, and without it ``fetch_models`` returns the static
``FALLBACK_MODELS`` list - the Rust binary owns live fetching, so the plugin
never performs direct in-process HTTP catalog discovery.

Hermes-native diagnostics:
``cloudflare_doctor()``, ``cloudflare_setup()``,
``cloudflare_catalog_refresh()``, ``cloudflare_catalog_export()``,
``cloudflare_model_inspect()`` plus the
``CLI_COMMANDS`` dispatch table and ``cloudflare_command()`` router. Each
command delegates to the ``auth-cloudflare`` executable (``doctor --format
json``, ``catalog refresh --format json``, ``catalog export <fmt>``,
``model inspect <id> --format json``) when the binary is present and
compatible; without the binary, ``doctor`` and ``model inspect`` return
token-redacting Python-side results computed from the environment /
``MODEL_POLICY`` / ``FALLBACK_MODELS``, while ``catalog refresh`` and
``catalog export`` fail closed (the Rust binary owns live fetching). Every
subprocess call is timeout-bound, and stderr is never echoed verbatim - only
the error class and command name are surfaced, so a misconfigured binary can
never leak a credential through diagnostics output.

When this module is imported inside the Hermes CLI runtime the same four
commands are wired as ``hermes cloudflare <cmd>`` through the SUPPORTED
plugin extension point ``PluginContext.register_cli_command``
(``hermes_cli/plugins.py``), which ``hermes_cli/main.py``
``_register_plugin_cli_commands`` attaches to the argparse tree. In
bare-provider contexts (tests, probes, embedded imports) that wiring is
skipped and the commands stay available through ``cloudflare_command()`` /
``CLI_COMMANDS``. If the Hermes CLI hook were ever removed, the exact file
needing a new hook is ``hermes_cli/main.py`` (``_register_plugin_cli_commands``,
which reads ``PluginManager._cli_commands`` after ``discover_plugins()``).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

API_BASE = "https://api.cloudflare.com/client/v4"
# Legacy Hermes-compatible env names (primary), Auth-Cloudflare canonical
# names (fallback) - the Rust core owns the resolution precedence.
TOKEN_ENV = "CLOUDFLARE_API_TOKEN"
ACCOUNT_ENV = "CLOUDFLARE_ACCOUNT_ID"
AUTH_TOKEN_ENV = "AUTH_CLOUDFLARE_API_TOKEN"
AUTH_ACCOUNT_ENV = "AUTH_CLOUDFLARE_ACCOUNT_ID"
# The `*_BASE_URL` suffix is what stock Hermes
# (hermes_cli/auth._register_plugin_provider) uses to detect the base-URL env
# var and map it to ProviderConfig.base_url_env_var, which the setup wizard
# then pre-fills. 'hermes cloudflare setup' writes the account-derived URL
# here - never the user.
BASE_URL_ENV = "CLOUDFLARE_BASE_URL"
DEFAULT_MODEL = "@cf/deepseek-ai/deepseek-v4-flash-0731"

# Model policy + agent-model lists: GENERATED data from
# fixtures/plugin_defaults.json, refreshed by scripts/regenerate-fixtures.py
# against the live Cloudflare Workers AI catalog. The Rust core owns the
# canonical policy via the `auth-cloudflare policy get --format json` bridge;
# this file is its regenerated mirror, not a hand-maintained list. Status
# priority: recommended < available < experimental < hidden; `rank` orders
# models within a status; `default` marks the sole development default;
# hidden models (e.g. safety classifiers) are never primary-agent options.
# Loading is DEFENSIVE: any read/parse/validation failure logs a warning and
# falls back to a minimal built-in set so plugin import never crashes.


def _load_generated_defaults() -> tuple[
    str, dict[str, dict[str, object]], tuple[str, ...], tuple[str, ...]
]:
    """Load DEFAULT_MODEL / MODEL_POLICY / PRIMARY_AGENT_MODELS /
    FALLBACK_MODELS from the generated fixtures/plugin_defaults.json.

    Guarded (same pattern as _read_protocol_version / _read_binary_version):
    any failure - missing file, bad JSON, missing keys, wrong types - falls
    back to a minimal built-in set with a warning; import never raises.
    """
    minimal_policy: dict[str, dict[str, object]] = {
        "@cf/deepseek-ai/deepseek-v4-flash-0731": {
            "status": "recommended",
            "rank": 10,
            "default": True,
            "primary_agent_eligible": True,
            "reason": "Validated development default.",
        },
    }
    minimal: tuple[str, ...] = ("@cf/deepseek-ai/deepseek-v4-flash-0731",)
    try:
        path = Path(__file__).resolve().parent / "fixtures" / "plugin_defaults.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("plugin_defaults.json is not an object")
        default_model = data.get("default_model")
        if not isinstance(default_model, str) or not default_model:
            raise TypeError(
                "plugin_defaults.json 'default_model' is not a non-empty string"
            )
        model_policy = data.get("model_policy")
        if not isinstance(model_policy, dict) or not all(
            isinstance(mid, str) and isinstance(entry, dict)
            for mid, entry in model_policy.items()
        ):
            raise TypeError(
                "plugin_defaults.json 'model_policy' is not an id->entry object"
            )
        primary = data.get("primary_agent_models")
        if not isinstance(primary, list) or not all(
            isinstance(mid, str) and mid for mid in primary
        ):
            raise TypeError(
                "plugin_defaults.json 'primary_agent_models' is not a list of ids"
            )
        fallback = data.get("fallback_models")
        if not isinstance(fallback, list) or not all(
            isinstance(mid, str) and mid for mid in fallback
        ):
            raise TypeError(
                "plugin_defaults.json 'fallback_models' is not a list of ids"
            )
    except Exception as exc:  # noqa: BLE001 - degrade, never crash import
        try:
            from providers.base import logger
        except Exception:
            import logging

            logger = logging.getLogger(__name__)  # type: ignore[assignment]
        logger.warning(
            "auth-hermes-cloudflare: could not load fixtures/plugin_defaults.json (%s) - "
            "using minimal built-in defaults; run scripts/regenerate-fixtures.py",
            exc,
        )
        return (
            "@cf/deepseek-ai/deepseek-v4-flash-0731",
            minimal_policy,
            minimal,
            minimal,
        )
    return default_model, model_policy, tuple(primary), tuple(fallback)


DEFAULT_MODEL, MODEL_POLICY, PRIMARY_AGENT_MODELS, FALLBACK_MODELS = (
    _load_generated_defaults()
)

# Cloudflare OpenAI-compatible reasoning-effort wire vocabulary. The Workers AI
# chat-completions schema documents ``reasoning_effort`` as the enum
# low|medium|high (per-model docs, e.g. deepseek-v4-flash-0731) - narrower than
# Hermes' internal ladder (none..ultra) and than the generic OpenAI-compat wire
# set (none..max). build_api_kwargs_extras clamps onto this set; "none" and
# disabled efforts omit the field entirely (Cloudflare has no "none" level and
# no ``thinking`` toggle on this surface).
CLOUDFLARE_REASONING_EFFORTS: tuple[str, ...] = ("low", "medium", "high")

# The plugin's vision-capable model: the only primary-agent-eligible Cloudflare
# model that accepts image input (llama-3.2-11b vision). default_vision_model()
# returns it so Hermes' auxiliary vision calls route here instead of the
# text-only main model (which 400s on image content).
CLOUDFLARE_VISION_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

# Executable bridge: the auth-cloudflare CLI owns catalog
# and policy; the plugin only discovers it, handshakes, and consumes JSON.
# Discovery NEVER downloads at import - downloading is the installer's job
# (download.sh) and happens only when Hermes activates the provider.
BINARY_NAME = "auth-cloudflare"
_PLUGIN_DIR = Path(__file__).resolve().parent


def _read_protocol_version() -> int:
    """The JSON CLI protocol version from the sibling PROTOCOL_VERSION file.

    Guarded: any read/parse failure falls back to 1 so plugin import never
    breaks on a missing or malformed file.
    """
    try:
        raw = (_PLUGIN_DIR / "PROTOCOL_VERSION").read_text(encoding="utf-8").strip()
        if raw:
            return int(raw)
    except Exception:
        pass
    return 1


PROTOCOL_VERSION = _read_protocol_version()

# The catalog schema the plugin can consume - mirrors the Rust binary's
# CATALOG_SCHEMA_VERSION (schema.rs). A binary reporting a version JSON
# without this schema version is rejected by the handshake.
CATALOG_SCHEMA_VERSION = 1


def _read_binary_version() -> str:
    """The minimum binary package version from the sibling BINARY_VERSION file.

    Guarded: any read/parse failure falls back to "0.0.0" so plugin import
    never breaks on a missing or malformed file (and no minimum is enforced).
    """
    try:
        raw = (_PLUGIN_DIR / "BINARY_VERSION").read_text(encoding="utf-8").strip()
        if raw:
            return raw
    except Exception:
        pass
    return "0.0.0"


BINARY_VERSION = _read_binary_version()

# FALLBACK_MODELS: the generated non-hidden chat-model list from
# fixtures/plugin_defaults.json (default first, then recommended, available,
# experimental; safety/classifier models like llama-guard-3-8b excluded) -
# loaded at the top of this module by _load_generated_defaults().

# Non-chat modalities + safety classifiers filtered from the primary picker.
_NON_CHAT_FRAGMENTS = (
    "embed",
    "image",
    "audio",
    "video",
    "speech",
    "tts",
    "rerank",
    "guard",
    "classifier",
    "segment",
    "whisper",
    "translation",
    "m2m",
    "imagen",
    "flux",
    "stable-diffusion",
)

# MODEL_POLICY status priority for ordering: recommended < available <
# experimental; models without a policy entry sort after every
# policy-classified model ("the rest").
_STATUS_PRIORITY = {"recommended": 0, "available": 1, "experimental": 2}


def _policy_sort_key(mid: str, index: int) -> tuple[int, int, int]:
    """Order *mid* by MODEL_POLICY: (status priority, rank, stable index).

    recommended < available < experimental, then models absent from
    MODEL_POLICY. Hidden/safety models must be excluded by callers before
    sorting.
    """
    entry = MODEL_POLICY.get(mid)
    if not isinstance(entry, dict):
        return (3, 1_000_000, index)
    status = entry.get("status")
    priority = _STATUS_PRIORITY.get(status, 3) if isinstance(status, str) else 3
    rank = entry.get("rank")
    if isinstance(rank, bool) or not isinstance(rank, (int, float, str)):
        rank = 1_000_000
    else:
        try:
            rank = int(rank)
        except (TypeError, ValueError):
            rank = 1_000_000
    return (priority, rank, index)


def _env(*names: str) -> str | None:
    """First non-empty env var among *names* (canonical then legacy order)."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def account_id() -> str | None:
    """The configured account id (AUTH_CLOUDFLARE_* then CLOUDFLARE_*), or None."""
    return _env(AUTH_ACCOUNT_ENV, ACCOUNT_ENV)


def api_token() -> str | None:
    """The configured API token (never printed), or None."""
    return _env(AUTH_TOKEN_ENV, TOKEN_ENV)


def inference_base_url() -> str:
    """OpenAI-compatible inference base URL for the configured account.

    Prefers an explicit ``CLOUDFLARE_BASE_URL`` override (the value
    ``hermes cloudflare setup`` writes to ``~/.hermes/.env``; also read by
    stock Hermes' ``_provider_env_base_url`` at runtime), then derives from
    the account ID.
    """
    override = os.getenv(BASE_URL_ENV, "").strip()
    if override:
        return override.rstrip("/")
    return f"{API_BASE}/accounts/{account_id() or '<ACCOUNT_ID>'}/ai/v1"


def catalog_url() -> str | None:
    """Catalog endpoint; None when no account is configured."""
    aid = account_id()
    if not aid:
        return None
    return f"{API_BASE}/accounts/{aid}/ai/models/search?format=openrouter&per_page=1000"


def runtime_cache_binary_path() -> Path:
    """Plugin-local runtime cache fallback (download.sh installs to ~/.hermes/bin)."""
    return _PLUGIN_DIR / "binaries" / BINARY_NAME


def locate_auth_cloudflare_binary() -> str | None:
    """Locate the auth-cloudflare executable; NEVER downloads at import.

    Order: ``AUTH_CLOUDFLARE_BIN`` env > ``auth-cloudflare`` on PATH >
    ``~/.hermes/bin/auth-cloudflare`` > plugin ``bin/`` > plugin runtime
    cache (``binaries/``). Downloading belongs to the installer
    (``download.sh``) and only happens when Hermes activates the provider
    or the user runs setup - never during module import.
    """
    explicit = os.getenv("AUTH_CLOUDFLARE_BIN", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    on_path = shutil.which(BINARY_NAME)
    if on_path:
        return on_path
    for candidate in (
        Path.home() / ".hermes" / "bin" / BINARY_NAME,
        _PLUGIN_DIR / "bin" / BINARY_NAME,
        runtime_cache_binary_path(),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _valid_semver(value: object) -> bool:
    """True when *value* is a semver-shaped ``x.y.z`` string (optional prerelease)."""
    return isinstance(value, str) and bool(
        re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-.][0-9A-Za-z.]+)?", value.strip())
    )


def _semver_tuple(value: str) -> tuple[int, int, int] | None:
    """Parse a semver ``x.y.z`` string into ``(major, minor, patch)`` ints.

    Prerelease/build suffixes are ignored for ordering; returns None when
    *value* is not semver-shaped.
    """
    if not _valid_semver(value):
        return None
    try:
        major, minor, rest = value.strip().split(".", 2)
        patch = rest.split("-", 1)[0].split("+", 1)[0]
        return (int(major), int(minor), int(patch))
    except (TypeError, ValueError):
        return None


def check_binary_compatibility(bin_path: str) -> tuple[bool, str]:
    """Version handshake: run ``auth-cloudflare version --format json``.

    Validates executable presence, JSON validity, ``name == BINARY_NAME``,
    ``catalog_schema_versions`` containing ``CATALOG_SCHEMA_VERSION``,
    ``protocol_version >= PROTOCOL_VERSION``, a parseable ``package_version``,
    and ``package_version >= BINARY_VERSION``. Returns ``(ok, detail)``;
    ``detail`` is actionable and token-free - stderr is NEVER echoed verbatim
    because a misconfigured binary could print secrets to it (only the JSON
    parse error and the command name are surfaced).
    """
    try:
        proc = subprocess.run(
            [bin_path, "version", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError:
        return False, f"auth-cloudflare binary not found at {bin_path!r}"
    except subprocess.TimeoutExpired:
        return False, "auth-cloudflare version --format json timed out after 5s"
    except OSError as exc:
        return False, f"auth-cloudflare binary could not be executed: {exc}"
    if proc.returncode != 0:
        return False, (
            f"auth-cloudflare version exited with code {proc.returncode} - run "
            "`auth-hermes-cloudflare upgrade` or "
            "`cargo install auth-cloudflare --locked --force`"
        )
    try:
        info = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return False, (
            f"auth-cloudflare version returned invalid JSON ({exc}); expected "
            "`version --format json` output"
        )
    if not isinstance(info, dict):
        return False, "auth-cloudflare version JSON is not an object"
    name = info.get("name") or info.get("binary")
    if name != BINARY_NAME:
        return False, f"unexpected binary name {name!r}; expected {BINARY_NAME!r}"
    schema_versions = info.get("catalog_schema_versions")
    if (
        not isinstance(schema_versions, list)
        or CATALOG_SCHEMA_VERSION not in schema_versions
    ):
        return False, (
            f"auth-cloudflare catalog_schema_versions {schema_versions!r} does not "
            f"include the supported schema version {CATALOG_SCHEMA_VERSION} - run "
            "`auth-hermes-cloudflare upgrade`"
        )
    try:
        protocol = int(info.get("protocol_version", 0))
    except (TypeError, ValueError):
        return False, (
            f"auth-cloudflare protocol_version {info.get('protocol_version')!r} "
            "is not an integer"
        )
    if protocol < PROTOCOL_VERSION:
        return False, (
            f"auth-cloudflare protocol {protocol} is older than the required "
            f"{PROTOCOL_VERSION} - run `auth-hermes-cloudflare upgrade`"
        )
    package_version = info.get("package_version")
    if not _valid_semver(package_version):
        return False, (
            f"auth-cloudflare package_version {package_version!r} is not valid semver"
        )
    actual = _semver_tuple(package_version)
    required = _semver_tuple(BINARY_VERSION)
    if actual is not None and required is not None and actual < required:
        return False, (
            f"auth-cloudflare package_version {package_version} is older than the "
            f"required {BINARY_VERSION} - run `auth-hermes-cloudflare upgrade`"
        )
    return True, f"auth-cloudflare {package_version} (protocol {protocol}) compatible"


def _fetch_models_via_binary(bin_path: str, timeout: float = 15.0) -> list[str] | None:
    """``auth-cloudflare catalog get --format json`` -> eligible model ids.

    The executable is the single source of truth for policy: only
    ``primary_agent_eligible`` models are returned, ``hidden`` status is
    excluded, and ordering is recommended first, experimental after (stable
    within a group). Returns None on any failure so fetch_models falls back
    to the static FALLBACK_MODELS list.
    """
    try:
        proc = subprocess.run(
            [bin_path, "catalog", "get", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        from providers.base import logger

        logger.debug(
            "fetch_models(%s): auth-cloudflare catalog get: %s", BINARY_NAME, exc
        )
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return None
    models = data.get("models") if isinstance(data, dict) else data
    if not isinstance(models, list):
        return None
    eligible: list[tuple[tuple[int, int, int], int, str]] = []
    for index, item in enumerate(models):
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        if not item.get("primary_agent_eligible"):
            continue
        if item.get("status") == "hidden":
            continue
        # Keep the binary's primary_agent_eligible gate; reorder by
        # MODEL_POLICY (recommended first in rank order, hidden/safety out).
        eligible.append((_policy_sort_key(mid, index), index, mid))
    eligible.sort()
    return [mid for _, _, mid in eligible] or None


# ── Hermes-native diagnostics ─────────────────────────
# Every command delegates to the auth-cloudflare binary when it is present and
# compatible; Python-side fallbacks are token-redacting and only cover doctor
# / model inspect (catalog refresh + export fail closed without the binary).
# Security invariants: every subprocess call has a timeout, stderr is NEVER
# echoed verbatim (only error class + command name), and no output contains
# the API token or a full account id.

DOCTOR_TIMEOUT = 20.0
CATALOG_TIMEOUT = 30.0
INSPECT_TIMEOUT = 15.0
_BINARY_MISSING_MSG = "auth-cloudflare binary not found; run download.sh"
_EXPORT_FORMATS = ("yaml", "markdown")


def _redact_account_id(value: str | None) -> str | None:
    """First 6 + "..." + last 4 of *value*; None stays None.

    The account id is not a secret (the token is), but doctor output and
    setup diagnostics keep it redacted like the Rust core's
    ``doctor --format json`` contract. Short values are fully masked to
    avoid overlapping prefix/suffix gibberish.
    """
    if not value:
        return None
    value = value.strip()
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:6]}...{value[-4:]}"


def _run_binary(bin_path: str, args: list[str], timeout: float) -> tuple[int, str, str]:
    """Run ``auth-cloudflare <args>``; returns ``(rc, stdout, stderr)``.

    Synthetic rc codes for launch failures (no stderr is captured into the
    caller's error text): 127 = not found, 124 = timeout, 126 = could not
    execute. stdout/stderr are otherwise returned verbatim - redaction of
    *reported* errors happens in the callers, never by trimming here.
    """
    try:
        proc = subprocess.run(
            [bin_path, *args], capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError:
        return 127, "", ""
    except subprocess.TimeoutExpired:
        return 124, "", ""
    except OSError:
        return 126, "", ""
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def _binary_failure_msg(command: str, rc: int) -> str:
    """Token-free failure text: error class + command name only."""
    if rc == 127:
        return f"auth-cloudflare {command}: binary not found"
    if rc == 124:
        return f"auth-cloudflare {command}: timed out"
    if rc == 126:
        return f"auth-cloudflare {command}: could not be executed"
    return f"auth-cloudflare {command}: exited with code {rc}"


def _run_binary_json(
    bin_path: str, args: list[str], timeout: float
) -> tuple[dict | None, str | None, int | None]:
    """``auth-cloudflare <args>`` -> ``(parsed_json, error, exit_code)``.

    ``error`` is token-free (class + command name); ``exit_code`` is None on
    a launch failure (synthetic rc), the process rc otherwise. The parsed
    payload is only returned as a ``dict`` - anything else is an error.
    """
    rc, stdout, _stderr = _run_binary(bin_path, args, timeout)
    if rc in (127, 124, 126):
        return None, _binary_failure_msg(" ".join(args), rc), None
    if rc != 0:
        # The binary prints a full, token-free diagnostic JSON even on
        # non-zero exit (e.g. doctor with a missing account id exits 2
        # with configured:false detail). Relay it instead of discarding
        # it so callers see exactly what is missing; keep rc for
        # deterministic exit-code propagation.
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                data.setdefault("exit_code", rc)
                return data, None, rc
        except json.JSONDecodeError:
            pass
        return None, _binary_failure_msg(" ".join(args), rc), rc
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return (
            None,
            f"auth-cloudflare {' '.join(args)} returned invalid JSON ({exc})",
            rc,
        )
    if not isinstance(data, dict):
        return (
            None,
            f"auth-cloudflare {' '.join(args)} returned a non-object payload",
            rc,
        )
    return data, None, rc


def _locate_usable_binary(binary: str | None) -> tuple[str | None, str | None]:
    """``(bin_path, incompat_detail)`` for diagnostics use.

    *binary* may be an explicit path; None auto-locates. ``(None, None)``
    means no binary is installed at all; ``(None, detail)`` means a binary
    exists but failed the version handshake (detail is actionable and
    token-free).
    """
    bin_path = binary if binary is not None else locate_auth_cloudflare_binary()
    if bin_path is None:
        return None, None
    ok, detail = check_binary_compatibility(bin_path)
    if not ok:
        return None, detail
    return bin_path, None


def cloudflare_doctor(binary: str | None = None) -> dict:
    """Doctor report: auth-cloudflare binary first, Python fallback second.

    With a compatible binary, ``auth-cloudflare doctor --format json`` is
    run (20s timeout) and its JSON is returned verbatim plus
    ``"source": "binary"`` - with defensive re-redaction if the binary ever
    returns the raw account id. Without a binary, a minimal environment
    report is computed in Python with ``"source": "python-fallback"``: the
    API token is only ever reported as ``value_redacted: True`` and the
    account id is redacted ``first6...last4``. No paid inference request is
    ever made.
    """
    bin_path, incompat = _locate_usable_binary(binary)
    if bin_path is not None:
        data, error, rc = _run_binary_json(
            bin_path, ["doctor", "--format", "json"], timeout=DOCTOR_TIMEOUT
        )
        if data is None:
            return {
                "status": "error",
                "error": error or _binary_failure_msg("doctor", rc or 1),
                "source": "binary",
                "exit_code": rc,
            }
        result = dict(data)
        # Defensive redaction: never trust the binary's redaction blindly.
        raw_aid = account_id()
        if raw_aid:
            aid = result.get("account_id")
            if isinstance(aid, dict) and aid.get("redacted") == raw_aid:
                aid["redacted"] = _redact_account_id(raw_aid)
            endpoint = result.get("endpoint")
            if isinstance(endpoint, dict):
                url = endpoint.get("base_url")
                if isinstance(url, str) and raw_aid in url:
                    endpoint["base_url"] = url.replace(
                        raw_aid, _redact_account_id(raw_aid) or "<redacted>"
                    )
        result.setdefault("source", "binary")
        return result
    if incompat is not None:
        return {
            "status": "error",
            "error": incompat,
            "source": "binary",
            "exit_code": 3,
        }

    aid = account_id()
    token = api_token()
    if aid and token:
        status = "ok"
    elif aid or token:
        status = "warning"
    else:
        status = "error"
    return {
        "status": status,
        "account_id": {
            "configured": bool(aid),
            "redacted": _redact_account_id(aid),
        },
        "api_token": {"configured": bool(token), "value_redacted": True},
        "endpoint": {
            "base_url": (
                inference_base_url().replace(aid, _redact_account_id(aid))
                if aid
                else None
            )
        },
        "catalog_cache": {"present": False},
        "source": "python-fallback",
        "note": _BINARY_MISSING_MSG,
    }


def cloudflare_catalog_refresh(binary: str | None = None) -> dict:
    """Live catalog refresh - delegated to the auth-cloudflare binary ONLY.

    ``auth-cloudflare catalog refresh --format json`` (30s timeout) and the
    JSON is returned as-is. There is intentionally NO Python fallback: the
    Rust binary owns live fetching, so without it this fails closed with
    exit_code 3.
    """
    bin_path, incompat = _locate_usable_binary(binary)
    if bin_path is None:
        return {
            "status": "error",
            "error": incompat or _BINARY_MISSING_MSG,
            "exit_code": 3,
        }
    data, error, rc = _run_binary_json(
        bin_path, ["catalog", "refresh", "--format", "json"], timeout=CATALOG_TIMEOUT
    )
    if data is None:
        return {
            "status": "error",
            "error": error or _binary_failure_msg("catalog refresh", rc or 1),
            "exit_code": rc,
        }
    return data


def cloudflare_catalog_export(fmt: str, binary: str | None = None) -> dict:
    """Export the catalog as ``yaml`` or ``markdown`` via the binary.

    ``auth-cloudflare catalog export <fmt>`` (30s timeout); the stdout is
    returned as ``content``. Fails closed without a compatible binary
    (exit_code 3) - there is no Python-side exporter.
    """
    fmt = (fmt or "").strip().lower()
    if fmt not in _EXPORT_FORMATS:
        return {
            "status": "error",
            "error": f"unsupported export format {fmt!r}; expected yaml or markdown",
            "exit_code": 2,
        }
    bin_path, incompat = _locate_usable_binary(binary)
    if bin_path is None:
        return {
            "status": "error",
            "error": incompat or _BINARY_MISSING_MSG,
            "exit_code": 3,
        }
    rc, stdout, _stderr = _run_binary(
        bin_path, ["catalog", "export", fmt], timeout=CATALOG_TIMEOUT
    )
    if rc != 0:
        return {
            "status": "error",
            "error": _binary_failure_msg("catalog export", rc),
            "exit_code": rc,
        }
    return {"status": "ok", "format": fmt, "content": stdout}


# Provider keys under which the sync writes Hermes `model_overrides`: the
# plugin provider id and the custom-provider path (both reach Hermes' model
# capability resolution - models_dev `_provider_override_section`).
_SYNC_PROVIDER_KEYS = ("auth-cloudflare-workers-ai", "cloudflare")


def _model_overrides_from_records(records: list[dict]) -> dict:
    """Map `models sync` capability records → Hermes `model_overrides` dict.

    Pure mapper (no binary, no config): each primary-agent-eligible record
    becomes ``model_overrides.<provider>.<model>.<field>`` under both provider
    keys. Only fields Hermes' override schema reads are emitted
    (context_window, supports_tools, supports_reasoning, model_family); a
    record with no usable fields is skipped entirely.
    """
    overrides: dict[str, dict[str, dict[str, object]]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        mid = record.get("id")
        if not isinstance(mid, str) or not mid:
            continue
        entry: dict[str, object] = {}
        context = record.get("context_window")
        if isinstance(context, int) and context > 0:
            entry["context_window"] = context
        if record.get("tool_call") is True:
            entry["supports_tools"] = True
        if record.get("reasoning") is True:
            entry["supports_reasoning"] = True
        if record.get("supports_vision") is True:
            entry["supports_vision"] = True
        family = record.get("model_family")
        if isinstance(family, str) and family:
            entry["model_family"] = family
        if not entry:
            continue
        for provider in _SYNC_PROVIDER_KEYS:
            overrides.setdefault(provider, {})[mid] = entry
    return overrides


def cloudflare_models_sync(
    binary: str | None = None, dry_run: bool = False, **kwargs: Any
) -> dict:
    """Apply the live catalog as Hermes `model_overrides` - no core edits.

    The auth-cloudflare binary generates the capability records (`models sync
    --format json`, cache-first like `catalog get`); this function maps them
    onto ``model_overrides.<provider>.<model>.<field>`` and writes the section
    through Hermes' supported config path (``hermes_cli.config.save_config``
    with ``merge_existing`` - the same write API plugins' post_setup uses).
    ``dry_run`` prints the dict without writing. Fails closed without a
    compatible binary; degrades to a warning when ``hermes_cli.config`` is not
    importable (bare-provider context) instead of raising.
    """
    bin_path, incompat = _locate_usable_binary(binary)
    if bin_path is None:
        return {
            "status": "error",
            "error": incompat or _BINARY_MISSING_MSG,
            "exit_code": 3,
        }
    data, error, rc = _run_binary_json(
        bin_path, ["models", "sync", "--format", "json"], timeout=CATALOG_TIMEOUT
    )
    if data is None:
        return {
            "status": "error",
            "error": error or _binary_failure_msg("models sync", rc or 1),
            "exit_code": rc,
        }
    records = data.get("models") if isinstance(data, dict) else None
    if not isinstance(records, list):
        return {
            "status": "error",
            "error": "models sync returned no models array",
            "exit_code": 3,
        }
    overrides = _model_overrides_from_records(records)
    model_count = len(records)
    if dry_run:
        return {
            "status": "ok",
            "dry_run": True,
            "source": data.get("source"),
            "cache_status": data.get("cache_status"),
            "model_count": model_count,
            "model_overrides": overrides,
        }
    try:
        from hermes_cli.config import load_config_readonly, save_config
    except Exception:
        return {
            "status": "warning",
            "error": "hermes_cli.config not importable - run `hermes cloudflare models sync` "
            "from a Hermes session to apply",
            "model_count": model_count,
            "model_overrides": overrides,
            "exit_code": 0,
        }
    # Skip the write when the config already carries these exact overrides
    # (session-start self-heal would otherwise rewrite config.yaml every turn).
    try:
        existing_raw = load_config_readonly().get("model_overrides") or {}
        existing = existing_raw if isinstance(existing_raw, dict) else {}
        merged: dict[str, dict] = {**existing}
        for provider, models in overrides.items():
            merged.setdefault(provider, {}).update(models)
        if merged == existing:
            return {
                "status": "ok",
                "unchanged": True,
                "source": data.get("source"),
                "cache_status": data.get("cache_status"),
                "model_count": model_count,
            }
    except Exception:  # noqa: BLE001 - degrade to the safe merge write
        pass
    try:
        save_config({"model_overrides": overrides}, merge_existing=True)
    except Exception as exc:  # noqa: BLE001 - degrade, never raise
        return {
            "status": "error",
            "error": f"could not write model_overrides: {exc}",
            "exit_code": 1,
        }
    return {
        "status": "ok",
        "source": data.get("source"),
        "cache_status": data.get("cache_status"),
        "provider_keys": list(overrides.keys()),
        "model_count": model_count,
        "entries_written": sum(len(v) for v in overrides.values()),
    }


# ── Deep Hermes integration (Aphrodite-style hooks; no core edits) ──────────
# Registered at import time through the same PluginContext reach-in as the CLI
# commands (_try_register_hermes_cli_command). Each registration is guarded and
# idempotent; outside the Hermes CLI runtime everything degrades to a no-op so
# the plugin imports cleanly on stock core / bare-provider contexts.

_CLOUDFLARE_HOST_FRAGMENT = "api.cloudflare.com"


def _is_cloudflare_call(context: dict) -> bool:
    """True when *context* names the Cloudflare provider or base URL."""
    provider = str(context.get("provider") or "").strip().lower()
    if provider in ("auth-cloudflare-workers-ai", "cloudflare", "custom:cloudflare"):
        return True
    host = str(context.get("base_url") or "")
    return _CLOUDFLARE_HOST_FRAGMENT in host


def _cloudflare_llm_request_middleware(payload: dict, **context: Any) -> dict:
    """Rewrite the outgoing LLM request onto Cloudflare's OpenAI-compatible schema.

    Runs inside Hermes' ``llm_request`` middleware chain (the ``{"request":
    {...}}`` result REPLACES the provider kwargs before the API call):

    - ``max_tokens`` → ``max_completion_tokens`` (the Cloudflare schema marks
      ``max_tokens`` deprecated in favour of ``max_completion_tokens``)
    - ``reasoning_effort`` clamped to the documented low|medium|high enum
    - ``temperature`` clamped to the documented 0..2 range

    Only returns a replacement when something actually changed and only for
    Cloudflare calls; every other request passes through untouched.
    """
    if not _is_cloudflare_call(context):
        return {}
    request = payload.get("request")
    if not isinstance(request, dict):
        return {}
    rewritten = dict(request)
    changed = False
    max_tokens = rewritten.get("max_tokens")
    if (
        isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens > 0
    ):
        rewritten.pop("max_tokens", None)
        rewritten["max_completion_tokens"] = max_tokens
        changed = True
    effort = rewritten.get("reasoning_effort")
    if isinstance(effort, str) and effort.strip():
        lowered = effort.strip().lower()
        if lowered not in CLOUDFLARE_REASONING_EFFORTS:
            clamped = _clamp_effort(lowered, CLOUDFLARE_REASONING_EFFORTS)
            if clamped != effort:
                rewritten["reasoning_effort"] = clamped
                changed = True
    temperature = rewritten.get("temperature")
    if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
        bounded = max(0.0, min(2.0, float(temperature)))
        if bounded != temperature:
            rewritten["temperature"] = bounded
            changed = True
    if not changed:
        return {}
    return {
        "request": rewritten,
        "source": "auth-hermes-cloudflare",
        "reason": "cloudflare OpenAI-compatible schema",
    }


def _cloudflare_api_error_classification(**kwargs: Any) -> dict | None:
    """Cloudflare-specific API error classification (Hermes' FailoverReason).

    Maps Cloudflare's error surface onto the failover vocabulary: 429 →
    rate_limit (backoff + rotate), 503/529 → overloaded (backoff), 500/502 →
    server_error (retry), 401/403 → auth (refresh/rotate), 400 → invalid
    request (not retryable - the message hints the cause, e.g. an invalid
    reasoning_effort). Returns None for non-Cloudflare calls so the default
    classifier applies unchanged.
    """
    if not _is_cloudflare_call(kwargs):
        return None
    try:
        status = (
            int(kwargs.get("status_code"))
            if kwargs.get("status_code") is not None
            else 0
        )
    except (TypeError, ValueError):
        status = 0
    try:
        from agent.error_classifier import FailoverReason
    except Exception:
        return None
    if status == 429:
        return {
            "reason": FailoverReason.rate_limit.name,
            "retryable": True,
            "should_rotate_credential": True,
            "message": "Cloudflare rate limit (429)",
        }
    if status in (503, 529):
        return {
            "reason": FailoverReason.overloaded.name,
            "retryable": True,
            "message": "Cloudflare model overloaded"
            if status == 529
            else "Cloudflare service unavailable (503)",
        }
    if status in (500, 502):
        return {
            "reason": FailoverReason.server_error.name,
            "retryable": True,
            "message": "Cloudflare server error",
        }
    if status in (401, 403):
        return {
            "reason": FailoverReason.auth.name,
            "retryable": True,
            "should_rotate_credential": True,
            "message": "Cloudflare token rejected (401/403)",
        }
    if status == 400:
        invalid_request = getattr(FailoverReason, "invalid_request", None)
        if invalid_request is None:
            return None
        return {
            "reason": invalid_request.name,
            "retryable": False,
            "message": "Cloudflare rejected the request (400) - check reasoning_effort / params",
        }
    return None


def _cloudflare_on_session_start(**kwargs: Any) -> None:
    """Session-start self-heal: re-apply model_overrides from the live catalog.

    Cache-first (no network when the cache is fresh) and write-free when the
    config is already current - the same self-heal pattern as the Aphrodite
    plugin's startup layout check. Never raises.
    """
    try:
        result = cloudflare_models_sync()
        if result.get("status") == "error":
            from providers.base import logger

            logger.debug(
                "auth-hermes-cloudflare: session-start models sync skipped: %s",
                result.get("error"),
            )
    except Exception:
        from providers.base import logger

        logger.debug(
            "auth-hermes-cloudflare: session-start models sync failed", exc_info=True
        )


_HOOKS_REGISTERED = False


def _try_register_hermes_hooks_and_middleware() -> None:
    """Wire the llm_request middleware + API hooks into Hermes (guarded, idempotent)."""
    global _HOOKS_REGISTERED
    if _HOOKS_REGISTERED:
        return
    try:
        from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
    except Exception:
        return
    try:
        manager = get_plugin_manager()
        context = PluginContext(
            PluginManifest(name="auth-hermes-cloudflare", key="auth-hermes-cloudflare"),
            manager,
        )
        context.register_middleware("llm_request", _cloudflare_llm_request_middleware)
        context.register_hook(
            "transform_api_error_classification", _cloudflare_api_error_classification
        )
        context.register_hook("on_session_start", _cloudflare_on_session_start)
        _HOOKS_REGISTERED = True
    except Exception as exc:  # noqa: BLE001 - degrade, never break import
        from providers.base import logger

        logger.warning(
            "auth-hermes-cloudflare: hook/middleware registration failed (%s) - "
            "API-call rewriting and error classification are DISABLED for this session",
            exc,
        )


def _model_policy() -> dict:
    """MODEL_POLICY if the sibling task defined it, else ``{}``.

    MODEL_POLICY landed in a parallel change; this module works with or
    without it (``globals()`` resolves at call time, so definition order is
    irrelevant).
    """
    policy = globals().get("MODEL_POLICY")
    return policy if isinstance(policy, dict) else {}


def cloudflare_model_inspect(model_id: str, binary: str | None = None) -> dict:
    """Inspect one model: binary first, MODEL_POLICY/FALLBACK_MODELS second.

    With a compatible binary, ``auth-cloudflare model inspect <id> --format
    json`` (15s timeout) is returned as-is. Without the binary, the model is
    looked up in MODEL_POLICY (if present) and FALLBACK_MODELS and reported
    as ``{id, in_policy, status, default, primary_agent_eligible, reason}``;
    an unknown id returns ``{id, found: false}``.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        return {
            "status": "error",
            "error": "model inspect requires a model id",
            "exit_code": 2,
        }
    bin_path, incompat = _locate_usable_binary(binary)
    if bin_path is not None:
        data, error, rc = _run_binary_json(
            bin_path,
            ["model", "inspect", model_id, "--format", "json"],
            timeout=INSPECT_TIMEOUT,
        )
        if data is None:
            return {
                "status": "error",
                "error": error or _binary_failure_msg("model inspect", rc or 1),
                "exit_code": rc,
            }
        return data

    policy = _model_policy()
    entry = policy.get(model_id) if isinstance(policy, dict) else None
    in_fallback = model_id in FALLBACK_MODELS
    if entry is None and not in_fallback:
        return {"id": model_id, "found": False}

    is_default = model_id == DEFAULT_MODEL
    if isinstance(entry, dict):
        entry_default = entry.get("default")
        if isinstance(entry_default, bool):
            is_default = entry_default
        status = entry.get("status")
        status = status if isinstance(status, str) and status else None
        eligible = entry.get("primary_agent_eligible")
        if not isinstance(eligible, bool):
            eligible = True
        reason = entry.get("reason")
        reason = reason if isinstance(reason, str) and reason else None
    else:
        status = None
        eligible = True
        reason = None

    return {
        "id": model_id,
        "in_policy": True,
        "status": status
        or ("recommended" if model_id == FALLBACK_MODELS[0] else "available"),
        "default": is_default,
        "primary_agent_eligible": eligible,
        "reason": reason
        or (
            "Validated development default."
            if is_default
            else "Model is in the plugin's policy/fallback catalog."
        ),
    }


CLI_COMMANDS: dict[str, object] = {
    "doctor": cloudflare_doctor,
    # NOTE: "setup" is intentionally NOT in this table - cloudflare_setup is
    # defined later in the module (after validate_setup), so referencing it
    # here would raise NameError at import. The cloudflare_command() router
    # and the argparse handler dispatch "setup" at call time instead.
    "catalog refresh": cloudflare_catalog_refresh,
    "catalog export": cloudflare_catalog_export,
    "model inspect": cloudflare_model_inspect,
    "models sync": cloudflare_models_sync,
}


def cloudflare_command(cmd: str, **kwargs) -> dict:
    """Route a diagnostic command string to ``CLI_COMMANDS``.

    Accepted surface: ``doctor``, ``setup``,
    ``catalog refresh``, ``catalog export <yaml|markdown>``,
    ``model inspect <model-id>``.
    Unknown or malformed commands return a usage error dict (exit_code 2) -
    never raise. Extra keyword arguments (e.g. ``binary=``) pass through to
    the target command.
    """
    parts = (cmd or "").split()
    if not parts:
        return {
            "status": "error",
            "error": "usage: cloudflare doctor | setup | catalog refresh | "
            "catalog export <yaml|markdown> | model inspect <model-id>",
            "exit_code": 2,
        }
    head = parts[0].lower()
    if head == "doctor":
        return cloudflare_doctor(**kwargs)
    if head == "setup":
        return cloudflare_setup(**kwargs)
    if head == "catalog":
        if len(parts) < 2:
            return {
                "status": "error",
                "error": "usage: cloudflare catalog refresh | catalog export <yaml|markdown>",
                "exit_code": 2,
            }
        if parts[1] == "refresh":
            return cloudflare_catalog_refresh(**kwargs)
        if parts[1] == "export":
            if len(parts) < 3:
                return {
                    "status": "error",
                    "error": "cloudflare catalog export requires a format (yaml or markdown)",
                    "exit_code": 2,
                }
            return cloudflare_catalog_export(parts[2], **kwargs)
        return {
            "status": "error",
            "error": f"unknown catalog subcommand {parts[1]!r}",
            "exit_code": 2,
        }
    if head == "model":
        if len(parts) < 2 or parts[1] != "inspect":
            return {
                "status": "error",
                "error": "usage: cloudflare model inspect <model-id>",
                "exit_code": 2,
            }
        if len(parts) < 3:
            return {
                "status": "error",
                "error": "cloudflare model inspect requires a model id",
                "exit_code": 2,
            }
        return cloudflare_model_inspect(" ".join(parts[2:]), **kwargs)
    if head == "models":
        if len(parts) < 2 or parts[1] != "sync":
            return {
                "status": "error",
                "error": "usage: cloudflare models sync [--dry-run]",
                "exit_code": 2,
            }
        return cloudflare_models_sync(**kwargs)
    return {
        "status": "error",
        "error": f"unknown cloudflare command {head!r}",
        "exit_code": 2,
    }


# ── hermes cloudflare <cmd> CLI wiring (deliverable 6) ───────────────────────
# The SUPPORTED extension point is ``PluginContext.register_cli_command``
# (hermes_cli/plugins.py); hermes_cli/main.py::_register_plugin_cli_commands
# attaches every ``PluginManager._cli_commands`` entry as a top-level
# ``hermes <name>`` subparser. Registration is guarded + idempotent so a
# bare providers import (tests/probes) costs nothing.


def _cloudflare_cli_emit(result: dict) -> int:
    """Print a diagnostics result as JSON; error dicts become exit codes."""
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("status") == "error":
        rc = result.get("exit_code")
        return int(rc) if isinstance(rc, int) else 1
    return 0


def _cloudflare_cli_doctor(args) -> int:  # noqa: ARG001 - argparse namespace
    return _cloudflare_cli_emit(cloudflare_doctor())


def _cloudflare_cli_setup(args) -> int:  # noqa: ARG001 - argparse namespace
    return _cloudflare_cli_emit(cloudflare_setup())


def _cloudflare_cli_catalog_refresh(args) -> int:  # noqa: ARG001
    return _cloudflare_cli_emit(cloudflare_catalog_refresh())


def _cloudflare_cli_catalog_export(args) -> int:
    result = cloudflare_catalog_export(args.format)
    if result.get("status") == "error":
        return _cloudflare_cli_emit(result)
    print(result.get("content", ""), end="")
    return 0


def _cloudflare_cli_model_inspect(args) -> int:
    return _cloudflare_cli_emit(cloudflare_model_inspect(args.model_id))


def _cloudflare_cli_models_sync(args) -> int:
    return _cloudflare_cli_emit(cloudflare_models_sync(dry_run=args.dry_run))


def _cloudflare_cli_bare(args) -> int:  # noqa: ARG001
    print("Auth Cloudflare Workers AI diagnostics")
    print(
        "usage: hermes cloudflare doctor | setup | catalog refresh | "
        "catalog export {yaml,markdown} | model inspect <model-id> | models sync [--dry-run]"
    )
    return 0


def _build_cloudflare_cli_parser(subparser) -> None:
    """Build the ``hermes cloudflare`` argparse tree (doctor/setup/catalog/model)."""
    sub = subparser.add_subparsers(
        dest="cloudflare_cmd",
        metavar="{doctor,setup,catalog,model}",
        help="Auth Cloudflare Workers AI diagnostics",
    )

    p_doctor = sub.add_parser(
        "doctor", help="Provider/environment diagnostic report (JSON)"
    )
    p_doctor.set_defaults(func=_cloudflare_cli_doctor)

    p_setup = sub.add_parser(
        "setup",
        help="Write the account-derived base URL to ~/.hermes/.env "
        "(CLOUDFLARE_BASE_URL) and validate setup",
    )
    p_setup.set_defaults(func=_cloudflare_cli_setup)

    p_catalog = sub.add_parser(
        "catalog", help="Catalog operations: refresh (live) or export (yaml|markdown)"
    )
    cat = p_catalog.add_subparsers(dest="catalog_cmd", metavar="{refresh,export}")
    p_refresh = cat.add_parser(
        "refresh", help="Live-refresh the catalog through the auth-cloudflare binary"
    )
    p_refresh.set_defaults(func=_cloudflare_cli_catalog_refresh)
    p_export = cat.add_parser("export", help="Export the catalog as yaml or markdown")
    p_export.add_argument(
        "format", choices=_EXPORT_FORMATS, help="export format (yaml|markdown)"
    )
    p_export.set_defaults(func=_cloudflare_cli_catalog_export)

    p_model = sub.add_parser("model", help="Model operations: inspect")
    m = p_model.add_subparsers(dest="model_cmd", metavar="{inspect}")
    p_inspect = m.add_parser("inspect", help="Inspect one model (policy/metadata)")
    p_inspect.add_argument(
        "model_id", help="model id, e.g. @cf/deepseek-ai/deepseek-v4-flash-0731"
    )
    p_inspect.set_defaults(func=_cloudflare_cli_model_inspect)

    p_models = sub.add_parser("models", help="Model catalog operations: sync")
    ms = p_models.add_subparsers(dest="models_cmd", metavar="{sync}")
    p_sync = ms.add_parser(
        "sync", help="Apply the live catalog as Hermes model_overrides config"
    )
    p_sync.add_argument(
        "--dry-run",
        action="store_true",
        help="print the overrides without writing config",
    )
    p_sync.set_defaults(func=_cloudflare_cli_models_sync)


def _try_register_hermes_cli_command() -> None:
    """Wire ``hermes cloudflare <cmd>`` inside the Hermes CLI runtime.

    No-op (cheaply) in bare-provider contexts where ``hermes_cli`` is not
    importable or where the command is already registered (the module can be
    imported twice - once via the providers registry and once via
    PluginManager discovery). Any failure degrades to a debug log: the
    programmatic ``cloudflare_command()`` dispatcher remains the API.
    """
    try:
        from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
    except Exception:
        return
    try:
        manager = get_plugin_manager()
        if "cloudflare" in manager._cli_commands:
            return
        context = PluginContext(
            PluginManifest(name="auth-hermes-cloudflare", key="auth-hermes-cloudflare"),
            manager,
        )
        context.register_cli_command(
            "cloudflare",
            help="Auth Cloudflare Workers AI diagnostics "
            "(doctor, catalog refresh/export, model inspect)",
            description="Provider diagnostics delegated to the auth-cloudflare binary "
            "when present; token-redacting Python fallbacks otherwise.",
            setup_fn=_build_cloudflare_cli_parser,
            handler_fn=_cloudflare_cli_bare,
        )
    except Exception as exc:
        from providers.base import logger

        logger.debug(
            "auth-hermes-cloudflare: hermes cloudflare CLI registration skipped: %s",
            exc,
        )


def _clamp_effort(effort: str, supported: tuple[str, ...]) -> str:
    """Nearest-WEAKER clamp of *effort* onto *supported*; never escalates cost.

    Prefers the canonical ``agent.reasoning_effort.clamp_effort`` when the
    Hermes runtime is importable (stock core), with a local ladder fallback so
    the plugin still clamps correctly in bare-provider/test contexts.
    """
    try:
        from agent.reasoning_effort import clamp_effort

        clamped = clamp_effort(effort, supported)
        if isinstance(clamped, str) and clamped:
            return clamped
    except Exception:
        pass
    ladder = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
    if effort in supported:
        return effort
    try:
        idx = ladder.index(effort)
    except ValueError:
        return effort  # bespoke name - pass through unchanged
    weaker = [lvl for lvl in supported if lvl in ladder and ladder.index(lvl) < idx]
    return max(weaker, key=ladder.index) if weaker else min(supported, key=ladder.index)


class CloudflareProfile(ProviderProfile):
    """Cloudflare Workers AI profile with LAZY, account-aware URLs.

    Plugin discovery runs before the profile .env is loaded into os.environ,
    so eager ``base_url=compute()`` at import time bakes the ``<ACCOUNT_ID>``
    placeholder and every runtime read 404s. These properties compute from
    ``os.environ`` at ACCESS time; the absorbing setters swallow the parent
    dataclass ``__init__`` assignments.
    """

    @property
    def base_url(self) -> str:
        return inference_base_url()

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url_override = value

    @property
    def models_url(self) -> str:
        return catalog_url() or ""

    @models_url.setter
    def models_url(self, value: str) -> None:
        self._models_url_override = value

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Live catalog: auth-cloudflare binary first, static fallback second.

        When the auth-cloudflare executable is present and compatible, the
        catalog comes from ``catalog get --format json`` (policy-ordered,
        primary-agent-eligible ids only). Without a binary, or on any binary
        catalog error, the static ``FALLBACK_MODELS`` list is returned - the
        Rust binary owns live fetching, so there is no direct in-process HTTP
        catalog discovery.
        """
        bin_path = locate_auth_cloudflare_binary()
        if bin_path is not None:
            ok, detail = check_binary_compatibility(bin_path)
            if ok:
                models = _fetch_models_via_binary(bin_path, timeout=15.0)
                if models is not None:
                    return models
                from providers.base import logger

                logger.debug(
                    "fetch_models(%s): binary catalog failed; static fallback",
                    self.name,
                )
            else:
                from providers.base import logger

                logger.debug(
                    "fetch_models(%s): binary incompatible - %s", self.name, detail
                )
        return list(FALLBACK_MODELS)

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        """Tri-state effort vocabulary for *model* on the Cloudflare wire.

        Plugin chat models (PRIMARY_AGENT_MODELS / FALLBACK_MODELS): the
        documented low|medium|high set. Everything else (non-chat, safety,
        unknown ids): () so no reasoning field is ever sent to an endpoint
        that does not document one.
        """
        mid = (model or "").strip()
        if mid in PRIMARY_AGENT_MODELS or mid in FALLBACK_MODELS:
            return CLOUDFLARE_REASONING_EFFORTS
        return ()

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Clamp Hermes' effort onto Cloudflare's low|medium|high wire set.

        Cloudflare's OpenAI-compatible schema takes ``reasoning_effort``
        top-level (enum low|medium|high) and has no ``thinking`` toggle and no
        "none" level: a disabled / missing / "none" effort omits the field
        entirely. The generic transport clamp (OPENAI_COMPAT_WIRE_EFFORTS,
        tops out at ``max``) would forward levels Cloudflare rejects, so this
        profile clamps again onto the documented set (#89503 class).
        """
        supported = self.supported_reasoning_efforts(model)
        if (
            not supported
            or not isinstance(reasoning_config, dict)
            or reasoning_config.get("enabled") is False
        ):
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if not effort or effort == "none":
            return {}, {}
        return {}, {"reasoning_effort": _clamp_effort(effort, supported)}

    def default_vision_model(self) -> str | None:
        """The plugin's vision-capable model (llama-3.2-11b-vision).

        The main agent model is text-only, so Hermes' auxiliary vision calls
        must route here - otherwise they hit the text model and the image
        input 400s (auxiliary_client `_resolve_provider_vision_default`
        consults this hook).
        """
        return CLOUDFLARE_VISION_MODEL


def validate_setup() -> dict:
    """Six-step setup validation.

    Order: (1) account_id present + valid 32-hex shape, (2) api_token
    present (its value is NEVER printed), (3) catalog discovery succeeds,
    (4) at least one usable chat-agent model, (5) the DeepSeek V4 Flash
    development default is in the catalog, (6) OPTIONAL explicit test
    inference - skipped by default. Validation never performs a paid
    inference request. Returns ``{"overall": "ok"|"warning"|"error",
    "steps": [...]}``.
    """
    steps: list[dict] = []

    # 1. Account ID exists and has a valid shape (Cloudflare: 32 hex chars).
    aid = account_id()
    if not aid:
        steps.append({"step": "account_id", "ok": False, "detail": "missing"})
    elif not re.fullmatch(r"[0-9a-fA-F]{32}", aid):
        steps.append(
            {
                "step": "account_id",
                "ok": False,
                "detail": f"invalid shape ({len(aid)} chars, expected 32 hex)",
            }
        )
    else:
        # Redacted like `doctor`: prefix...suffix, never the full value.
        steps.append(
            {
                "step": "account_id",
                "ok": True,
                "detail": f"configured ({aid[:6]}...{aid[-4:]})",
            }
        )

    # 2. API token exists; the value never appears in any detail.
    token = api_token()
    steps.append(
        {
            "step": "api_token",
            "ok": bool(token),
            "detail": "configured" if token else "missing",
        }
    )

    # 3. Catalog discovery succeeds (fetch_models swallows most failures;
    #    the guard keeps this step exception-safe regardless).
    try:
        models = cloudflare.fetch_models()
    except Exception:
        models = None
        detail = "error: catalog discovery raised"
    else:
        detail = len(models) if models else "error: no models returned"
    steps.append({"step": "catalog", "ok": bool(models), "detail": detail})

    # 4. At least one usable chat-agent (primary-agent) model.
    usable = [m for m in (models or []) if m in PRIMARY_AGENT_MODELS]
    steps.append(
        {
            "step": "chat_model",
            "ok": bool(usable),
            "detail": (
                f"{len(usable)} primary-agent model(s)"
                if usable
                else "no usable chat-agent model in catalog"
            ),
        }
    )

    # 5. The development default (DeepSeek V4 Flash) is present.
    has_default = DEFAULT_MODEL in (models or [])
    steps.append(
        {
            "step": "default_model",
            "ok": has_default,
            "detail": (
                f"{DEFAULT_MODEL} present"
                if has_default
                else f"{DEFAULT_MODEL} missing from catalog"
            ),
        }
    )

    # 6. Optional explicit test inference - SKIPPED by default: a normal
    #    setup run must never send a paid inference request.
    steps.append(
        {
            "step": "test_inference",
            "ok": None,
            "detail": "optional - not auto-run",
        }
    )

    oks = [s["ok"] for s in steps if s["ok"] is not None]
    if all(oks):
        overall = "ok"
    elif any(oks):
        overall = "warning"
    else:
        overall = "error"
    return {"overall": overall, "steps": steps}


def _persist_env_value(key: str, value: str) -> bool:
    """Persist ``key=value`` to ``~/.hermes/.env``; True when written.

    Prefers stock Hermes ``hermes_cli.config.save_env_value`` (idempotent,
    validates the var name, publishes to ``os.environ``); falls back to a
    direct file edit in bare-provider contexts where ``hermes_cli`` is not
    importable. Never prints or echoes the value beyond the caller's own
    reporting.
    """
    try:
        from hermes_cli.config import save_env_value

        save_env_value(key, value)
        return True
    except Exception:
        pass
    env_path = Path.home() / ".hermes" / ".env"
    try:
        lines = (
            env_path.read_text(encoding="utf-8").splitlines()
            if env_path.exists()
            else []
        )
        line = f"{key}={value}"
        idx = next((i for i, l in enumerate(lines) if l.startswith(f"{key}=")), None)
        if idx is not None:
            lines[idx] = line
        else:
            lines.append(line)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return True
    except Exception:
        return False


def cloudflare_setup() -> dict:
    """One-command setup: persist the derived base URL to ``~/.hermes/.env``.

    Stock Hermes' wizard (``hermes_cli/model_setup_flows._model_flow_api_key_provider``)
    pre-fills its Base URL prompt from ``ProviderConfig.base_url_env_var`` -
    the env var the profile declares via ``BASE_URL_ENV`` - so writing the
    URL derived from ``CLOUDFLARE_ACCOUNT_ID`` there means the user never
    types a Base URL. Also publishes it to ``os.environ`` for the current
    process and finishes with a ``validate_setup()`` summary. No credentials
    are printed; the account id is redacted like ``doctor``.
    """
    aid = account_id()
    if not aid:
        return {
            "status": "error",
            "error": (
                "CLOUDFLARE_ACCOUNT_ID is not configured - add it to "
                "~/.hermes/.env and re-run (the API token belongs there too)"
            ),
            "exit_code": 1,
        }
    if not api_token():
        return {
            "status": "error",
            "error": (
                "CLOUDFLARE_API_TOKEN is not configured - add it to "
                "~/.hermes/.env and re-run"
            ),
            "exit_code": 1,
        }
    url = inference_base_url()
    written = _persist_env_value(BASE_URL_ENV, url)
    os.environ[BASE_URL_ENV] = url
    if not written:
        return {
            "status": "error",
            "error": f"could not persist {BASE_URL_ENV} to ~/.hermes/.env",
            "exit_code": 1,
        }
    return {
        "status": "ok",
        "base_url_env": BASE_URL_ENV,
        "base_url": url,
        "account_id": _redact_account_id(aid),
        "persisted": written,
        "setup": validate_setup(),
    }


# Module-level instance + registration - the exact contract every bundled
# provider follows (import side effect: profile joins the registry, so
# list_providers()/the model picker see it immediately). URLs are LAZY (see
# CloudflareProfile) - the <ACCOUNT_ID> placeholder only appears in contexts
# that have not loaded the profile .env yet.
cloudflare = CloudflareProfile(
    name="auth-cloudflare-workers-ai",
    aliases=(
        "auth-cloudflare",
        "cloudflare",
        "cloudflare-workers-ai",
        "workers-ai",
        "cf-workers-ai",
        "cf",
    ),
    display_name="Auth Cloudflare Workers AI",
    description=(
        "Auth Cloudflare Workers AI - direct OpenAI-compatible access to "
        "Cloudflare-hosted Workers AI models, with account-aware discovery"
    ),
    signup_url="https://dash.cloudflare.com/profile/api-tokens",
    env_vars=(TOKEN_ENV, ACCOUNT_ENV, BASE_URL_ENV),
    api_mode="chat_completions",
    auth_type="api_key",
    default_aux_model=DEFAULT_MODEL,
    fallback_models=FALLBACK_MODELS,
    # Cloudflare's /ai/v1 has no /models endpoint; health probing is disabled
    # and the prebuilt catalog is authoritative. The base URL is derived from
    # the account ID: BASE_URL_ENV is declared above so stock Hermes
    # (hermes_cli/auth._register_plugin_provider) maps it to
    # ProviderConfig.base_url_env_var, and 'hermes cloudflare setup' writes
    # the derived URL there - the setup wizard pre-fills it, never prompts
    # the user to type an override. URLs stay LAZY (see CloudflareProfile).
    supports_health_check=False,
)
# fixed_base_url: a legacy core flag (patched pre-2026-09-11 hermes-agent
# cores) the setup wizard honors to skip the Base URL prompt entirely; stock
# core has NO such field (`hermes update` wipes core patches). Never pass it
# as a constructor kwarg - that raises TypeError on stock core. Set it
# post-construction only when the base class declares it; the env-var
# mechanism (BASE_URL_ENV above) is the fallback on cores without the field.
if hasattr(ProviderProfile, "fixed_base_url"):
    cloudflare.fixed_base_url = True

register_provider(cloudflare)

# Hermes-native diagnostics CLI wiring: guarded, idempotent,
# and a no-op outside the Hermes CLI runtime - see _try_register_hermes_cli_command.
_try_register_hermes_cli_command()
# Deep integration (Aphrodite-style): llm_request middleware + API hooks -
# guarded, idempotent, no-op outside the CLI runtime.
_try_register_hermes_hooks_and_middleware()
