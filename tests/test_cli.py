"""Tests for CLI wiring, --once output, and rendering helpers."""

from __future__ import annotations

import pytest

from localtok.app import build_renderables, model_row_cells, status_line
from localtok.cli import build_parser, build_providers, main, run_once
from localtok.collector import Collector
from localtok.models import ModelRow, Snapshot, Status
from localtok.providers import DemoProvider, OllamaProvider, OpenAICompatProvider


def test_demo_flag_selects_only_demo_provider():
    args = build_parser().parse_args(["--demo"])
    providers = build_providers(args)
    assert len(providers) == 1
    assert isinstance(providers[0], DemoProvider)


def test_no_flags_defaults_to_ollama_and_openai():
    args = build_parser().parse_args([])
    providers = build_providers(args)
    kinds = {type(p) for p in providers}
    assert OllamaProvider in kinds
    assert OpenAICompatProvider in kinds


def test_explicit_ollama_url_only(monkeypatch):
    monkeypatch.delenv("LOCALTOK_OPENAI_URL", raising=False)
    args = build_parser().parse_args(["--ollama", "http://test-host:11434"])
    providers = build_providers(args)
    assert len(providers) == 1
    assert isinstance(providers[0], OllamaProvider)
    assert providers[0].base_url == "http://test-host:11434"


def test_env_var_configures_url(monkeypatch):
    monkeypatch.setenv("LOCALTOK_OLLAMA_URL", "http://env-host:11434")
    monkeypatch.delenv("LOCALTOK_OPENAI_URL", raising=False)
    args = build_parser().parse_args([])
    providers = build_providers(args)
    ollama = [p for p in providers if isinstance(p, OllamaProvider)][0]
    assert ollama.base_url == "http://env-host:11434"


def test_status_line_counts():
    snap = Snapshot(
        models=[
            ModelRow("a", "demo", Status.LOADED),
            ModelRow("b", "demo", Status.AVAILABLE),
        ],
        errors=["boom"],
    )
    line = status_line(snap)
    assert "2 models" in line
    assert "1 loaded" in line
    assert "error" in line


def test_model_row_cells_markup_status():
    cells = model_row_cells(ModelRow("a", "demo", Status.LOADED))
    assert cells[0] == "a"
    assert "loaded" in cells[2]
    assert "green" in cells[2]


def test_build_renderables_counts_rows():
    snap = DemoProvider().snapshot(t=0.0)
    data = build_renderables(snap)
    assert len(data["model_rows"]) == 3
    assert len(data["gpu_rows"]) == 1
    assert "models" in data["status"]


@pytest.mark.asyncio
async def test_run_once_renders_demo():
    collector = Collector([DemoProvider()], use_gpu=False)
    text = await run_once(collector)
    assert "demo-7b" in text
    assert "demo-70b" in text
    assert "node-a" in text
    assert "== Models ==" in text
    # markup must be stripped from plain-text output
    assert "[green]" not in text
    assert "[/]" not in text


def test_main_once_exits_zero(capsys):
    rc = main(["--demo", "--once", "--no-gpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "demo-7b" in out
