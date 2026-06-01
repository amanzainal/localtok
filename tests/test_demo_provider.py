"""Tests for the synthetic demo provider."""

from __future__ import annotations

import pytest

from localtok.models import Status
from localtok.providers.demo import DemoProvider


def test_snapshot_is_deterministic_for_fixed_t():
    provider = DemoProvider()
    a = provider.snapshot(t=0.0)
    b = provider.snapshot(t=0.0)
    assert [m.tokens_per_sec for m in a.models] == [m.tokens_per_sec for m in b.models]


def test_snapshot_shape_and_synthetic_names():
    provider = DemoProvider()
    snap = provider.snapshot(t=1.0)

    names = [m.name for m in snap.models]
    assert "demo-7b" in names
    assert "demo-70b" in names
    assert "example/base-model-v1" in names

    loaded = [m for m in snap.models if m.status == Status.LOADED]
    assert len(loaded) == 2
    assert all(m.tokens_per_sec is not None for m in loaded)
    assert all(m.vram_bytes is not None for m in loaded)

    assert len(snap.gpus) == 1
    assert snap.gpus[0].name == "node-a"
    assert snap.gpus[0].mem_total_bytes is not None


def test_tps_varies_over_time():
    provider = DemoProvider()
    early = provider.snapshot(t=0.0).models[0].tokens_per_sec
    later = provider.snapshot(t=3.0).models[0].tokens_per_sec
    assert early != later


@pytest.mark.asyncio
async def test_poll_returns_snapshot():
    provider = DemoProvider()
    result = await provider.poll(client=None)  # demo ignores the client
    assert result.errors == []
    assert len(result.models) == 3
    assert len(result.gpus) == 1
