"""GPU discovery via ``nvidia-smi``.

We shell out to ``nvidia-smi`` with ``--query-gpu`` in CSV-no-units form, which
is stable and easy to parse. If the binary is missing (no NVIDIA GPU, or a
non-NVIDIA box) we simply return no rows — the dashboard just won't show a GPU
section. Parsing is a pure function so tests feed it captured CSV text.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from localtok.models import GpuRow

MB = 1024 * 1024

_QUERY_FIELDS = [
    "index",
    "name",
    "memory.used",
    "memory.total",
    "utilization.gpu",
    "temperature.gpu",
]


def _to_float(token: str) -> Optional[float]:
    token = token.strip()
    if not token or token.lower() in {"n/a", "[n/a]", "[not supported]"}:
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_nvidia_smi_csv(text: str) -> list[GpuRow]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``.

    Each line is::

        0, Example GPU, 1234, 49140, 37, 55

    where memory columns are MiB and util/temp are integers. Malformed lines
    are skipped rather than raising.
    """
    rows: list[GpuRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        mem_used = _to_float(parts[2])
        mem_total = _to_float(parts[3])
        util = _to_float(parts[4])
        temp = _to_float(parts[5])
        rows.append(
            GpuRow(
                index=index,
                name=name,
                mem_used_bytes=int(mem_used * MB) if mem_used is not None else None,
                mem_total_bytes=int(mem_total * MB) if mem_total is not None else None,
                utilization_pct=util,
                temperature_c=temp,
            )
        )
    return rows


def nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def query_gpus(timeout: float = 2.0) -> list[GpuRow]:
    """Run ``nvidia-smi`` and return GPU rows, or ``[]`` if unavailable."""
    if not nvidia_smi_available():
        return []
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    return parse_nvidia_smi_csv(out.stdout)
