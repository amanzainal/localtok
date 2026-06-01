"""Command-line entry point.

Builds the provider list from flags + environment, then either launches the
Textual dashboard or, with ``--once``, prints a single plain-text snapshot
(handy for screenshots, CI, or piping). No personal data, no hardcoded hosts:
defaults are loopback and every address is overridable via flag or env var.

Environment variables (all optional):
    LOCALTOK_OLLAMA_URL        base url for Ollama       (default http://127.0.0.1:11434)
    LOCALTOK_OPENAI_URL        base url for OpenAI-compat (default http://127.0.0.1:8000)
    LOCALTOK_OPENAI_API_KEY    bearer token for OpenAI-compat (optional)
    LOCALTOK_INTERVAL          refresh seconds            (default 1.0)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Sequence

from localtok import __version__
from localtok.app import GPU_COLUMNS, MODEL_COLUMNS, build_renderables
from localtok.collector import Collector
from localtok.providers import DemoProvider, OllamaProvider, OpenAICompatProvider
from localtok.providers.base import Provider
from localtok.providers.ollama import DEFAULT_BASE_URL as OLLAMA_DEFAULT
from localtok.providers.openai_compat import DEFAULT_BASE_URL as OPENAI_DEFAULT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="localtok",
        description="htop for local LLM inference — live tokens/sec, VRAM and loaded models.",
    )
    p.add_argument("--demo", action="store_true", help="run with synthetic data; no server or GPU needed")
    p.add_argument("--ollama", metavar="URL", help=f"poll an Ollama server (default {OLLAMA_DEFAULT})")
    p.add_argument("--openai", metavar="URL", help=f"poll an OpenAI-compatible server (default {OPENAI_DEFAULT})")
    p.add_argument("--no-gpu", action="store_true", help="skip nvidia-smi GPU discovery")
    p.add_argument(
        "--interval",
        type=float,
        default=float(os.environ.get("LOCALTOK_INTERVAL", "1.0")),
        help="refresh interval in seconds (default 1.0)",
    )
    p.add_argument("--once", action="store_true", help="print one plain-text snapshot and exit (no TUI)")
    p.add_argument("--version", action="version", version=f"localtok {__version__}")
    return p


def build_providers(args: argparse.Namespace) -> list[Provider]:
    """Decide which providers to run from flags + env.

    In demo mode we only use the synthetic provider. Otherwise: an Ollama and
    an OpenAI-compatible provider are each enabled if (a) explicitly given a
    URL flag, (b) given a URL via env var, or (c) no provider flags were passed
    at all — in which case we fall back to the loopback defaults so the tool is
    useful out of the box.
    """
    if args.demo:
        return [DemoProvider()]

    providers: list[Provider] = []

    ollama_url = args.ollama or os.environ.get("LOCALTOK_OLLAMA_URL")
    openai_url = args.openai or os.environ.get("LOCALTOK_OPENAI_URL")

    no_explicit = ollama_url is None and openai_url is None
    if no_explicit:
        # Sensible default: try local Ollama + local OpenAI-compat.
        ollama_url = OLLAMA_DEFAULT
        openai_url = OPENAI_DEFAULT

    if ollama_url:
        providers.append(OllamaProvider(base_url=ollama_url))
    if openai_url:
        providers.append(OpenAICompatProvider(base_url=openai_url))
    return providers


def _format_table(title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render an ASCII table (markup stripped) for --once output."""
    import re

    def clean(cell: str) -> str:
        return re.sub(r"\[/?[^\]]*\]", "", str(cell))

    str_rows = [[clean(c) for c in r] for r in rows]
    widths = [len(c) for c in columns]
    for r in str_rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def fmt(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    lines = [title, fmt(columns), "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in str_rows)
    if not str_rows:
        lines.append("(none)")
    return "\n".join(lines)


async def run_once(collector: Collector) -> str:
    async with collector:
        snap = await collector.refresh()
    data = build_renderables(snap)
    out = [
        _format_table("== Models ==", MODEL_COLUMNS, data["model_rows"]),
        "",
        _format_table("== GPUs ==", GPU_COLUMNS, data["gpu_rows"]),
        "",
        data["status"].replace("[red]", "").replace("[/]", ""),
    ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    providers = build_providers(args)
    collector = Collector(providers, use_gpu=not args.no_gpu)

    if args.once:
        print(asyncio.run(run_once(collector)))
        return 0

    # Lazy import so --once / --version don't pay the Textual import cost.
    from localtok.app import LocaltokApp

    app = LocaltokApp(collector, interval=args.interval)
    app.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
