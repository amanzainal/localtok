"""Tests for nvidia-smi CSV parsing."""

from __future__ import annotations

from localtok.gpu import MB, parse_nvidia_smi_csv
from tests.conftest import load_fixture


def test_parse_two_gpus():
    rows = parse_nvidia_smi_csv(load_fixture("nvidia_smi.csv"))
    assert len(rows) == 2

    g0 = rows[0]
    assert g0.index == 0
    assert g0.name == "Example GPU A"
    assert g0.mem_used_bytes == 5120 * MB
    assert g0.mem_total_bytes == 49140 * MB
    assert g0.utilization_pct == 64.0
    assert g0.temperature_c == 57.0
    # mem_pct derived
    assert round(g0.mem_pct, 1) == round(100 * 5120 / 49140, 1)

    assert rows[1].index == 1


def test_parse_skips_blank_and_malformed_lines():
    text = "\n0, Example GPU A, 100, 200, 5, 40\nbroken line\n, , , \n"
    rows = parse_nvidia_smi_csv(text)
    assert len(rows) == 1
    assert rows[0].index == 0


def test_parse_handles_na_values():
    text = "0, Example GPU A, [N/A], 200, [Not Supported], 40"
    rows = parse_nvidia_smi_csv(text)
    assert len(rows) == 1
    assert rows[0].mem_used_bytes is None
    assert rows[0].utilization_pct is None
    assert rows[0].temperature_c == 40.0


def test_parse_empty_string():
    assert parse_nvidia_smi_csv("") == []
