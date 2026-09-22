# Auth-Hermes-Cloudflare ☁️ Hermes Plugin

> **Auth Cloudflare Workers AI model-provider plugin for Hermes Agent - pure Python provider + Rust core executable.**
> **Live account-aware catalog discovery, OpenAI-compatible inference, 22-model fallback.**

`auth-hermes-cloudflare` registers the `auth-cloudflare-workers-ai` provider
(display name **Auth Cloudflare Workers AI**) against Cloudflare's
OpenAI-compatible Workers AI surface. The account ID is injected from the environment, the catalog is
fetched from `/ai/models/search` in the OpenRouter format, and the provider
shows up in `hermes model` with zero `custom_providers` wiring. **All
endpoint/auth logic lives in the Rust core; Python is a thin provider.**

[![plugin](https://img.shields.io/static/v1?label=plugin&message=v0.0.4&color=purple)](plugin.yaml)
[![hermes](https://img.shields.io/static/v1?label=hermes&message=%E2%89%A50.16.0&color=blue)](https://github.com/NousResearch/hermes-agent)
[![license](https://img.shields.io/static/v1?label=license&message=CC0-1.0&color=lightgrey)](https://github.com/PlayForm/Auth-Cloudflare/blob/Current/LICENSE)

---

## Install ⚡

### One-command

```bash
git clone https://github.com/PlayForm/Auth-Hermes-Cloudflare.git
ln -s "$(pwd)/Auth-Hermes-Cloudflare" ~/.hermes/plugins/auth-hermes-cloudflare
hermes plugins enable auth-hermes-cloudflare
hermes
```

Then export the two env vars the provider reads:

```bash
export CLOUDFLARE_ACCOUNT_ID="<your account id>" # Workers & Pages → Overview
export CLOUDFLARE_API_TOKEN="<scoped token>"     # Account → Workers AI → Write
hermes model                                     # pick: Auth Cloudflare Workers AI
```

The account ID is operational metadata, not a secret. The API token **is** a
secret - scope it to **Account → Workers AI → Write** (some dashboard
versions label the same permission **Workers AI → Edit**) and nothing else.
Do not request DNS, Workers Scripts, R2, D1, KV, Pages, Zero Trust, or
account administration permissions.

The provider path is **pure Python** - no Rust toolchain, no binary download,
no compiled dependencies. The optional `download.sh` flow (the prebuilt
`auth-cloudflare` executable) is only needed for the diagnostics and
conformance commands, not for using Auth Cloudflare Workers AI in Hermes.

### What changes after install

After installing and launching Hermes once:

```
~/.hermes/
├── plugins/
│   └── auth-hermes-cloudflare → /path/to/Auth-Hermes-Cloudflare    ← symlink to this repo
└── profiles/<name>/
    └── plugins/
        └── auth-hermes-cloudflare → ~/.hermes/plugins/auth-hermes-cloudflare
```

The plugin also adds to your Hermes config:

```yaml
# Added automatically on enable
plugins:
    enabled:
        - auth-hermes-cloudflare

# Provider block (manual or via the setup wizard)
providers:
    cloudflare:
        api_key_env: CLOUDFLARE_API_TOKEN
        base_url: https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1
        api_mode: chat_completions
```

No proxy processes are launched - the provider runs in-process.

### Verify it's working

```bash
# In a Hermes session:
hermes model # pick: Auth Cloudflare Workers AI

# Or via CLI - token health check:
curl https://api.cloudflare.com/client/v4/user/tokens/verify \
	-H "Authorization: Bearer $CLOUD...OKEN"
# → {"success":true,"result":{"id":"...","status":"active",...}}
```

### Clean uninstall

```bash
hermes plugins disable auth-hermes-cloudflare
rm ~/.hermes/plugins/auth-hermes-cloudflare
```

---

## Architecture 🏗️

```
Python (provider registration)              Rust core (single source of truth)
  __init__.py       233L                     auth-cloudflare (executable)
    register_provider(cloudflare)             ├─ auth:    account/token → endpoints
    lazy base_url / models_url                ├─ catalog: ModelRecord, ModelRole, CapabilityState
    fetch_models() (OpenRouter fallback)      ├─ cache:   account-scoped cache slug
                                              └─ policy:  model policy + capability table
                                             auth-hermes-cloudflare (Rust crate)
                                             └─ re-exports core types for hooks/tools
```

The Rust core owns the canonical endpoint/auth/catalog/policy logic and ships
as a single executable, `auth-cloudflare`. The Python provider mirrors it
in-process so the picker and wizard work with or without the executable. URLs
are computed lazily from `os.environ` at access time, because plugin
discovery runs before the profile `.env` is loaded.

---

## Provider 🛠️

| Item            | Value                                                                                         |
| :-------------- | :-------------------------------------------------------------------------------------------- |
| Provider name   | `auth-cloudflare-workers-ai`                                                                  |
| Aliases         | `auth-cloudflare`, `cloudflare`, `cloudflare-workers-ai`, `workers-ai`, `cf-workers-ai`, `cf` |
| Display name    | `Auth Cloudflare Workers AI`                                                                  |
| API mode        | `chat_completions`                                                                            |
| Auth type       | `api_key`                                                                                     |
| Signup          | [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)      |
| Health check    | disabled - Cloudflare has no `/models` endpoint; token verify is used instead                 |
| Fixed base URL  | yes - derived from the account ID, the wizard never prompts                                   |
| Default model   | `@cf/deepseek-ai/deepseek-v4-flash-0731`                                                      |
| Fallback models | 26 chat models generated from fixtures/plugin_defaults.json by scripts/regenerate-fixtures.py |

---

## Configuration ⚙️

Two environment variables drive everything. The `AUTH_CLOUDFLARE_*` names
are canonical; the `CLOUDFLARE_*` names are the legacy Hermes-compatible
aliases - the Rust core owns the resolution precedence.

| Variable                                               | Role                                                      |
| :----------------------------------------------------- | :-------------------------------------------------------- |
| `CLOUDFLARE_ACCOUNT_ID` / `AUTH_CLOUDFLARE_ACCOUNT_ID` | account ID - operational metadata, not a secret           |
| `CLOUDFLARE_API_TOKEN` / `AUTH_CLOUDFLARE_API_TOKEN`   | API token - a secret, only ever sent as a `Bearer` header |

```bash
export CLOUDFLARE_ACCOUNT_ID="<your account id>"
export CLOUDFLARE_API_TOKEN="<scoped token>" # Account → Workers AI → Write
```

---

## Executable Flow 📦

The provider path is pure Python, so this is optional. The `auth-cloudflare`
executable backs the full command surface - `hermes cloudflare doctor /
catalog refresh / catalog export / model inspect / models sync`, catalog
caching, and the conformance commands (`model verify --suite smoke|tool-loop`, `model
health`). `download.sh` installs it from GitHub Releases:

**`Terminal`**

```sh
bash download.sh [version] [target-triple]
```

- Version auto-detection order: `BINARY_VERSION` → `Cargo.toml` (monorepo) →
  latest `Cloudflare/v*` GitHub release tag.
- Release tag convention: `Cloudflare/v<version>` - the `Build` workflow
  attaches the per-target archives (`aarch64`/`x86_64` macOS + Linux) on
  that tag.
- Installs `auth-cloudflare` into `~/.hermes/bin/auth-cloudflare` by default
  (override with `BINARY_DIR` or `AUTH_CLOUDFLARE_BIN`).
- Checksum-verified: the archive checksum must match the resolved sums source
  before extraction - the plugin tree's `SHA256SUMS.txt` (generated by the
  `Build` workflow; used when its `BINARY_VERSION` matches the version being
  installed) first, then the release's per-target
  `SHA256SUMS-<target>.txt`; if neither exists a loud warning is printed and
  verification is skipped (older releases stay installable). The extracted
  binary self-validates via `auth-cloudflare version --format json`.
- Atomic + fail-closed: installs via copy-then-rename through a temp file,
  with an `EXIT` trap that removes every temp artifact on any failure.

Discovery order (`locate_auth_cloudflare_binary`): `AUTH_CLOUDFLARE_BIN`
env → `PATH` → `~/.hermes/bin` → plugin `bin/` → plugin `binaries/`.

The `auth-hermes-cloudflare` binary (`cargo install
auth-hermes-cloudflare --locked`) manages the engine executable:

```sh
auth-hermes-cloudflare status                 # JSON: plugin + binary state
auth-hermes-cloudflare install [options]      # run download.sh, then doctor
auth-hermes-cloudflare upgrade [options]      # alias of install (re-run download.sh)
auth-hermes-cloudflare doctor                 # run 'auth-cloudflare doctor --format json'
auth-hermes-cloudflare uninstall              # remove ~/.hermes/bin/auth-cloudflare (alias: unlink)
auth-hermes-cloudflare link --source <path>   # symlink/copy <path> → ~/.hermes/bin/auth-cloudflare
```

`install` options: `--plugin-dir <dir>` (where to find `download.sh`),
`--binary <path>` (exact install path, sets `AUTH_CLOUDFLARE_BIN`), and
`--no-download` (print manual install instructions instead). `upgrade`
re-runs `download.sh` with the latest `BINARY_VERSION`, so updating is the
same atomic, checksum-verified flow as a fresh install.

### Fine-grained Hermes integration (`models sync`)

`hermes cloudflare models sync` hands the live catalog to Hermes through its
supported `model_overrides` config path - no Hermes core edits:

```sh
hermes cloudflare models sync [--dry-run]
```

The `auth-cloudflare` binary emits one capability record per
primary-agent-eligible model (`models sync --format json`, cache-first like
`catalog get`): context window from the live API, tool-calling and
reasoning flags, the documented `reasoning_effort` vocabulary
(low|medium|high) and model family. The plugin maps these onto
`model_overrides.<provider>.<model>.<field>` for both provider keys
(`auth-cloudflare-workers-ai` and `cloudflare`) and writes the section
through `hermes_cli.config.save_config` (merge_existing) - the same write
API plugins' `post_setup` uses. Hermes' model-capability resolution
(`get_model_capabilities`) then reports the real context windows (e.g.
1,310,720 for deepseek-v4-flash-0731), `supports_tools`/`supports_reasoning`
and `model_family` for every synced model on both provider paths. Re-run
after catalog changes; `--dry-run` prints the section without writing.

### Deep Hermes integration (Aphrodite-style hooks)

Registered at plugin import time through Hermes' `PluginContext` reach-in
(the same surface the `hermes cloudflare` CLI commands use) - guarded,
idempotent, and a no-op outside the Hermes CLI runtime:

- **`llm_request` middleware** - rewrites every outgoing Cloudflare call onto
  the official schema: `max_tokens` → `max_completion_tokens` (the schema
  marks `max_tokens` deprecated), `reasoning_effort` clamped to the
  documented low|medium|high enum, `temperature` clamped to the 0..2 range.
  Only fires for Cloudflare calls; every other request passes through
  untouched.
- **`transform_api_error_classification` hook** - maps Cloudflare status codes
  onto Hermes' failover vocabulary: 429 → `rate_limit` (backoff + rotate),
  503/529 → `overloaded`, 500/502 → `server_error`, 401/403 → `auth` (rotate),
  400 → invalid request (non-retryable). Non-Cloudflare errors keep the
  default classification.
- **`on_session_start` hook** - self-heal: re-applies `model_overrides` from
  the live catalog at session start (cache-first, write-free when the config
  is already current - the same pattern as the Aphrodite plugin's startup
  layout check).

The official Workers AI schemas (`.playform/schemas`) confirm the contract
this integration targets: standard OpenAI message envelope, no
`reasoning_content` in either direction (Cloudflare does not expose DeepSeek
reasoning text, so Hermes' reasoning-echo machinery is inert), and
`usage.completion_tokens_details.reasoning_tokens` for reasoning-token
accounting.

> [!NOTE]
>
> Without the executable the picker still works pure-Python through the
> in-process fallback, but the diagnostics and conformance commands are
> unavailable. `download.sh` fetches the checksum-verified `auth-cloudflare`
> release asset from the `Cloudflare/v<version>` GitHub release built by the
> `Build` workflow - no Rust toolchain required.

---

## Dev Install (Rust source)

```bash
git clone https://github.com/PlayForm/Auth-Cloudflare.git
cd Cloudflare
git submodule update --init --recursive
cargo build --release -p auth-cloudflare -p auth-hermes-cloudflare
# auth-cloudflare executable at target/release/auth-cloudflare
```

---

## Files

```
Auth-Hermes-Cloudflare/
├── __init__.py          ← 233-line Python provider (register_provider, lazy URLs)
├── plugin.yaml          ← model-provider manifest (env vars, min Hermes version)
├── download.sh          ← Prebuilt executable installer (checksum-verified, atomic)
├── BINARY_VERSION       ← Expected executable version
├── PROTOCOL_VERSION     ← JSON CLI protocol version
├── README.md            ← This file
└── .gitignore
```
