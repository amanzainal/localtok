"""Tests for the core data model and formatting helpers."""

from __future__ import annotations

from localtok.models import GpuRow, ModelRow, Status, human_bytes


def test_human_bytes_scales():
    assert human_bytes(0) == "0 B"
    assert human_bytes(512) == "512 B"
    assert human_bytes(1024) == "1.0 KB"
    assert human_bytes(1536) == "1.5 KB"
    assert human_bytes(1024 ** 3) == "1.0 GB"
    assert human_bytes(48 * 1024 ** 3) == "48.0 GB"


def test_human_bytes_none_and_negative():
    assert human_bytes(None) == "-"
    assert human_bytes(-1) == "-"


def test_model_row_human_properties():
    row = ModelRow(
        name="example-7b",
        provider="ollama",
        status=Status.LOADED,
        size_bytes=5 * 1024 ** 3,
        vram_bytes=5 * 1024 ** 3,
        tokens_per_sec=92.58,
    )
    assert row.size_h == "5.0 GB"
    assert row.vram_h == "5.0 GB"
    assert row.tps_h == "92.6"


def test_model_row_missing_values_render_dash():
    row = ModelRow(name="x", provider="openai")
    assert row.size_h == "-"
    assert row.vram_h == "-"
    assert row.tps_h == "-"


def test_status_str():
    assert str(Status.LOADED) == "loaded"
    assert str(Status.AVAILABLE) == "available"


def test_gpu_mem_pct():
    gpu = GpuRow(index=0, name="node-a", mem_used_bytes=24, mem_total_bytes=48)
    assert gpu.mem_pct == 50.0
    # no total -> None, no crash
    assert GpuRow(index=1, name="node-b").mem_pct is None
