"""Tests for the Ollama adapter — pure parsing against captured JSON."""

from __future__ import annotations

import httpx
import pytest

from localtok.models import Status
from localtok.providers.ollama import (
    OllamaProvider,
    merge_rows,
    parse_ps,
    parse_tags,
)
from tests.conftest import load_json_fixture


def test_parse_ps_extracts_vram_and_marks_loaded():
    rows = parse_ps(load_json_fixture("ollama_ps.json"), "ollama")
    assert len(rows) == 2
    first = rows[0]
    assert first.name == "example-7b:latest"
    assert first.status == Status.LOADED
    assert first.size_bytes == 5045231104
    assert first.vram_bytes == 5045231104
    assert "example" in first.detail and "Q4_K_M" in first.detail


def test_parse_tags_marks_available():
    rows = parse_tags(load_json_fixture("ollama_tags.json"), "ollama")
    assert len(rows) == 3
    assert all(r.status == Status.AVAILABLE for r in rows)
    assert {r.name for r in rows} == {
        "example-7b:latest",
        "example-code:latest",
        "example-tiny:latest",
    }
    # available rows carry size but no VRAM
    assert all(r.vram_bytes is None for r in rows)


def test_merge_prefers_loaded_and_dedupes():
    loaded = parse_ps(load_json_fixture("ollama_ps.json"), "ollama")
    available = parse_tags(load_json_fixture("ollama_tags.json"), "ollama")
    merged = merge_rows(loaded, available)

    # 3 distinct names total (2 loaded overlap with tags + 1 tags-only)
    names = [r.name for r in merged]
    assert len(names) == 3
    assert len(set(names)) == 3

    # loaded models sort first
    assert merged[0].status == Status.LOADED
    assert merged[1].status == Status.LOADED
    assert merged[2].status == Status.AVAILABLE
    assert merged[2].name == "example-tiny:latest"

    # the overlapping model keeps its VRAM (came from /api/ps)
    seven_b = next(r for r in merged if r.name == "example-7b:latest")
    assert seven_b.status == Status.LOADED
    assert seven_b.vram_bytes == 5045231104


def test_parse_handles_empty_payload():
    assert parse_ps({}, "ollama") == []
    assert parse_tags({"models": None}, "ollama") == []


@pytest.mark.asyncio
async def test_poll_against_mock_transport():
    ps = load_json_fixture("ollama_ps.json")
    tags = load_json_fixture("ollama_tags.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=ps)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(base_url="http://test-host:11434")
    async with httpx.AsyncClient(transport=transport) as client:
        result = await provider.poll(client)

    assert result.errors == []
    assert len(result.models) == 3
    assert result.models[0].status == Status.LOADED


@pytest.mark.asyncio
async def test_poll_records_errors_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(base_url="http://test-host:11434")
    async with httpx.AsyncClient(transport=transport) as client:
        result = await provider.poll(client)

    assert result.models == []
    assert len(result.errors) == 2  # both endpoints failed
