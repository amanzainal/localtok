"""Core data model shared across providers and the UI.

These dataclasses are the *only* contract between provider adapters and the
dashboard. A provider's job is to turn whatever its server returns into a list
of :class:`ModelRow` / :class:`GpuRow` objects. The UI never sees provider-
specific JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Status(str, Enum):
    """Lifecycle of a model as far as the dashboard cares."""

    LOADED = "loaded"      # resident in memory / VRAM, ready to serve
    AVAILABLE = "available"  # known to the server but not currently loaded
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # nicer table rendering
        return self.value


def human_bytes(n: Optional[int]) -> str:
    """Render a byte count as a short human string (e.g. ``4.7 GB``).

    ``None`` renders as ``-`` so missing data never crashes the table.
    """
    if n is None:
        return "-"
    if n < 0:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"  # pragma: no cover - unreachable guard


@dataclass
class ModelRow:
    """One model as shown in the dashboard table."""

    name: str
    provider: str
    status: Status = Status.UNKNOWN
    size_bytes: Optional[int] = None     # on-disk / parameter size
    vram_bytes: Optional[int] = None     # VRAM currently held by this model
    tokens_per_sec: Optional[float] = None
    detail: str = ""                     # free-form extra (quantization, etc.)

    @property
    def size_h(self) -> str:
        return human_bytes(self.size_bytes)

    @property
    def vram_h(self) -> str:
        return human_bytes(self.vram_bytes)

    @property
    def tps_h(self) -> str:
        if self.tokens_per_sec is None:
            return "-"
        return f"{self.tokens_per_sec:.1f}"


@dataclass
class GpuRow:
    """One physical GPU as reported by nvidia-smi (or synthetic in demo mode)."""

    index: int
    name: str
    mem_used_bytes: Optional[int] = None
    mem_total_bytes: Optional[int] = None
    utilization_pct: Optional[float] = None
    temperature_c: Optional[float] = None

    @property
    def mem_used_h(self) -> str:
        return human_bytes(self.mem_used_bytes)

    @property
    def mem_total_h(self) -> str:
        return human_bytes(self.mem_total_bytes)

    @property
    def mem_pct(self) -> Optional[float]:
        if not self.mem_total_bytes:
            return None
        return 100.0 * (self.mem_used_bytes or 0) / self.mem_total_bytes


@dataclass
class Snapshot:
    """Everything the dashboard needs for a single refresh tick."""

    models: list[ModelRow] = field(default_factory=list)
    gpus: list[GpuRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
