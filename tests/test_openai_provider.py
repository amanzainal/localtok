"""Tests for the generic OpenAI-compatible adapter."""

from __future__ import annotations

import httpx
import pytest

from localtok.models import Status
from localtok.providers.openai_compat import OpenAICompatProvider, parse_models
from tests.conftest import load_json_fixture


def test_parse_models_data_shape():
    rows = parse_models(load_json_fixture("openai_models.json"), "openai")
    assert len(rows) == 3
    assert all(r.status == Status.AVAILABLE for r in rows)
    # sorted alphabetically by id
    assert [r.name for r in rows] == [
        "example/base-model-v1",
        "example/embed-model-v1",
        "example/instruct-model-v1",
    ]


def test_parse_models_bare_list_and_strings():
    rows = parse_models(["example/a", "example/b"], "openai")
    assert [r.name for r in rows] == ["example/a", "example/b"]


def test_parse_models_empty():
    assert parse_models({"data": []}, "openai") == []
    assert parse_models({}, "openai") == []


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("LOCALTOK_OPENAI_API_KEY", "sk-test-placeholder")
    provider = OpenAICompatProvider(base_url="http://test-host:8000")
    assert provider._headers() == {"Authorization": "Bearer sk-test-placeholder"}


def test_no_api_key_means_no_auth_header(monkeypatch):
    monkeypatch.delenv("LOCALTOK_OPENAI_API_KEY", raising=False)
    provider = OpenAICompatProvider(base_url="http://test-host:8000")
    assert provider._headers() == {}


@pytest.mark.asyncio
async def test_poll_against_mock_transport():
    payload = load_json_fixture("openai_models.json")
    seen_auth = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth["value"] = request.headers.get("authorization")
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    provider = OpenAICompatProvider(base_url="http://test-host:8000", api_key="sk-x")
    async with httpx.AsyncClient(transport=transport) as client:
        result = await provider.poll(client)

    assert result.errors == []
    assert len(result.models) == 3
    assert seen_auth["value"] == "Bearer sk-x"


@pytest.mark.asyncio
async def test_poll_error_is_captured():
    transport = httpx.MockTransport(lambda req: httpx.Response(503))
    provider = OpenAICompatProvider(base_url="http://test-host:8000")
    async with httpx.AsyncClient(transport=transport) as client:
        result = await provider.poll(client)
    assert result.models == []
    assert len(result.errors) == 1
