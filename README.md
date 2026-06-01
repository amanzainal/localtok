# localtok

**htop for local LLM inference** — a terminal dashboard showing tokens/sec, VRAM
and loaded models for your local LLM servers.

```
== Models ==
Model                  Provider  Status     Size     VRAM     tok/s  Detail
---------------------  --------  ---------  -------  -------  -----  --------------
demo-7b                demo      loaded     4.7 GB   5.0 GB   92.2   example q4_k_m
demo-70b               demo      loaded     40.0 GB  42.0 GB  25.2   example q4_k_m
example/base-model-v1  demo      available  2.3 GB   -        -      example fp16

== GPUs ==
GPU  Name    Mem Used  Mem Total  Mem %  Util %  Temp
---  ------  --------  ---------  -----  ------  ----
0    node-a  47.0 GB   48.0 GB    98%    60%     58°C

3 models (2 loaded) · 1 GPU(s)
```

*(the live TUI is colorized and refreshes on a timer; the block above is the
plain-text `--once` snapshot)*

## Why it exists

When you run models locally, the questions are always the same: *what's loaded
right now, how much VRAM is it eating, and how fast is it going?* The answers
are scattered across `ollama ps`, `nvidia-smi`, and whatever your OpenAI-compatible
server exposes. `localtok` puts them in one always-on table — the way `htop`
does for processes. Point it at your servers, leave it running in a pane, and
glance over whenever you want to know the state of your local inference rig.

## Install

Requires Python 3.10+. Using [uv](https://docs.astral.sh/uv/):

```bash
git clone <your-fork-url> localtok
cd localtok
uv sync                 # install runtime deps
uv run localtok --demo  # try it with synthetic data, no server needed
```

Or install into the current environment with pip:

```bash
pip install -e .
localtok --demo
```

## Usage

```bash
# Synthetic data — runs anywhere, no LLM server or GPU required.
localtok --demo

# Auto-detect local servers (Ollama on :11434 + OpenAI-compatible on :8000)
# plus any NVIDIA GPUs via nvidia-smi.
localtok

# Point at specific servers.
localtok --ollama http://127.0.0.1:11434 --openai http://127.0.0.1:8000

# One-shot plain-text snapshot (great for scripts, CI, or a screenshot).
localtok --demo --once

# Skip GPU discovery, slow the refresh down.
localtok --no-gpu --interval 2.0
```

In the live dashboard: press **r** to refresh now, **q** to quit.

### Configuration

Everything is configurable by flag or environment variable. Copy
[`.env.example`](.env.example) to `.env` and edit (the `.env` file is
git-ignored). No secrets are ever hardcoded — an API key, if your server needs
one, is read from `LOCALTOK_OPENAI_API_KEY`.

| Env var | Flag | Default | Purpose |
| --- | --- | --- | --- |
| `LOCALTOK_OLLAMA_URL` | `--ollama URL` | `http://127.0.0.1:11434` | Ollama server |
| `LOCALTOK_OPENAI_URL` | `--openai URL` | `http://127.0.0.1:8000` | OpenAI-compatible server |
| `LOCALTOK_OPENAI_API_KEY` | — | _(none)_ | Optional bearer token |
| `LOCALTOK_INTERVAL` | `--interval N` | `1.0` | Refresh seconds |

## How it works

```
                 +-----------------------------+
   providers --> | Collector.refresh()         | --> Snapshot --> Textual UI
   (async poll)  |  - fan-out, concurrent      |     (models +    (two tables
                 |  - errors isolated per-prov |      gpus +       + status)
                 |  - + nvidia-smi GPU rows    |      errors)
                 +-----------------------------+
```

* **Providers** are small async adapters behind one interface (`Provider.poll`):
  * **Ollama** — `GET /api/ps` (loaded models, with real VRAM via `size_vram`)
    merged with `GET /api/tags` (the full local library).
  * **OpenAI-compatible** — `GET /v1/models`, the lowest common denominator that
    works with llama.cpp's server, vLLM, LM Studio, LocalAI and friends.
  * **Demo** — fabricated data for `--demo`.
* **GPU rows** come from `nvidia-smi --query-gpu=...`; if the binary is absent,
  that section is simply empty.
* The **Collector** polls every provider concurrently. A failing or offline
  server becomes a counted error in the status line — it never crashes the
  dashboard or blocks the others.
* The **Textual** app owns a refresh timer and repaints two `DataTable`s. The
  rendering logic is a pure function (`build_renderables`) so it's unit-tested
  without a terminal.

All parsing is split into pure functions tested against captured JSON/CSV
fixtures, so the whole suite runs with **no live server**:

```bash
uv run pytest -q
```

## Roadmap

- [ ] Live tok/s for real servers by tailing a streaming completion.
- [ ] Sparkline history per model (tok/s over the last N ticks).
- [ ] More adapters: llama.cpp native endpoints, vLLM metrics, TGI.
- [ ] AMD/Intel GPU support (`rocm-smi`, `xpu-smi`).
- [ ] Sort/filter keybindings and a per-model detail pane.
- [ ] Optional Prometheus exporter mode.

## License

[MIT](LICENSE) © The Authors.
