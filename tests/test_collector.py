"""Tests for the Collector aggregation logic."""

from __future__ import annotations

import httpx
import pytest

from localtok.collector import Collector
from localtok.models import ModelRow, Status
from localtok.providers.base import PollResult, Provider
from localtok.providers.demo import DemoProvider


class _FakeProvider(Provider):
    name = "fake"

    def __init__(self, result: PollResult | None = None, boom: bool = False):
        super().__init__("fake://host")
        self._result = result or PollResult()
        self._boom = boom

    async def poll(self, client):
        if self._boom:
            raise RuntimeError("kaboom")
        return self._result


@pytest.mark.asyncio
async def test_collector_merges_multiple_providers():
    p1 = _FakeProvider(PollResult(models=[ModelRow("a", "fake", Status.LOADED)]))
    p2 = _FakeProvider(PollResult(models=[ModelRow("b", "fake", Status.AVAILABLE)]))
    async with Collector([p1, p2], use_gpu=False) as c:
        snap = await c.refresh()
    assert {m.name for m in snap.models} == {"a", "b"}
    assert snap.errors == []


@pytest.mark.asyncio
async def test_collector_isolates_provider_failure():
    good = _FakeProvider(PollResult(models=[ModelRow("ok", "fake", Status.LOADED)]))
    bad = _FakeProvider(boom=True)
    async with Collector([good, bad], use_gpu=False) as c:
        snap = await c.refresh()
    # good provider still contributed
    assert [m.name for m in snap.models] == ["ok"]
    # bad provider became an error, not a crash
    assert any("kaboom" in e for e in snap.errors)


@pytest.mark.asyncio
async def test_collector_with_demo_provider_end_to_end():
    async with Collector([DemoProvider()], use_gpu=False) as c:
        snap = await c.refresh()
    assert len(snap.models) == 3
    assert len(snap.gpus) == 1  # demo provides its own synthetic GPU


@pytest.mark.asyncio
async def test_collector_owns_client_when_none_passed():
    async with Collector([DemoProvider()], use_gpu=False) as c:
        assert c._client is not None
    # client closed and cleared on exit
    assert c._client is None


@pytest.mark.asyncio
async def test_collector_uses_injected_client():
    client = httpx.AsyncClient()
    c = Collector([DemoProvider()], use_gpu=False, client=client)
    async with c:
        await c.refresh()
    # injected client must NOT be closed by the collector
    assert not client.is_closed
    await client.aclose()
