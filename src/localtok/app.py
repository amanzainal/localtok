"""The Textual dashboard.

A thin view over :class:`~localtok.collector.Collector`. It owns a refresh timer,
calls ``collector.refresh()`` on each tick, and repaints two tables — one for
models, one for GPUs — plus a status line. All the interesting logic lives in
the collector/providers; this file is just rendering.

:func:`build_renderables` is factored out as a pure function so tests can assert
on what the dashboard *would* draw for a given Snapshot without spinning up a
terminal.
"""

from __future__ import annotations

from typing import Iterable

from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Footer, Header, Static

from localtok.collector import Collector
from localtok.models import GpuRow, ModelRow, Snapshot, Status

MODEL_COLUMNS = ("Model", "Provider", "Status", "Size", "VRAM", "tok/s", "Detail")
GPU_COLUMNS = ("GPU", "Name", "Mem Used", "Mem Total", "Mem %", "Util %", "Temp")


def _status_markup(status: Status) -> str:
    color = {
        Status.LOADED: "green",
        Status.AVAILABLE: "yellow",
        Status.UNKNOWN: "dim",
    }.get(status, "white")
    return f"[{color}]{status}[/]"


def model_row_cells(row: ModelRow) -> tuple[str, ...]:
    return (
        row.name,
        row.provider,
        _status_markup(row.status),
        row.size_h,
        row.vram_h,
        row.tps_h,
        row.detail or "-",
    )


def gpu_row_cells(row: GpuRow) -> tuple[str, ...]:
    pct = row.mem_pct
    return (
        str(row.index),
        row.name,
        row.mem_used_h,
        row.mem_total_h,
        f"{pct:.0f}%" if pct is not None else "-",
        f"{row.utilization_pct:.0f}%" if row.utilization_pct is not None else "-",
        f"{row.temperature_c:.0f}°C" if row.temperature_c is not None else "-",
    )


def status_line(snap: Snapshot) -> str:
    loaded = sum(1 for m in snap.models if m.status == Status.LOADED)
    line = (
        f"{len(snap.models)} models ({loaded} loaded) · {len(snap.gpus)} GPU(s)"
    )
    if snap.errors:
        line += f" · [red]{len(snap.errors)} error(s)[/]"
    return line


def build_renderables(snap: Snapshot) -> dict[str, object]:
    """Pure helper: turn a Snapshot into the cell data the UI will paint.

    Returns a dict with ``model_rows`` and ``gpu_rows`` (lists of cell tuples)
    plus ``status``. Used directly by tests.
    """
    return {
        "model_rows": [model_row_cells(m) for m in snap.models],
        "gpu_rows": [gpu_row_cells(g) for g in snap.gpus],
        "status": status_line(snap),
    }


class LocaltokApp(App):
    """Live dashboard for local LLM servers."""

    CSS = """
    Screen { layout: vertical; }
    #models-title, #gpu-title { padding: 0 1; color: $accent; text-style: bold; }
    #status { padding: 0 1; color: $text-muted; }
    DataTable { height: auto; }
    #gpu-table { height: auto; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_now", "Refresh"),
    ]

    def __init__(self, collector: Collector, interval: float = 1.0) -> None:
        super().__init__()
        self.collector = collector
        self.interval = interval

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Static("Models", id="models-title")
            yield DataTable(id="model-table", zebra_stripes=True)
            yield Static("GPUs", id="gpu-title")
            yield DataTable(id="gpu-table", zebra_stripes=True)
            yield Static("", id="status")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "localtok"
        self.sub_title = "htop for local LLM inference"

        mt = self.query_one("#model-table", DataTable)
        mt.add_columns(*MODEL_COLUMNS)
        gt = self.query_one("#gpu-table", DataTable)
        gt.add_columns(*GPU_COLUMNS)

        await self.collector.__aenter__()
        await self.action_refresh_now()
        self.set_interval(self.interval, self.refresh_dashboard)

    async def on_unmount(self) -> None:
        await self.collector.__aexit__(None, None, None)

    async def action_refresh_now(self) -> None:
        await self.refresh_dashboard()

    async def refresh_dashboard(self) -> None:
        snap = await self.collector.refresh()
        self._paint(snap)

    def _paint(self, snap: Snapshot) -> None:
        data = build_renderables(snap)
        self._fill_table("#model-table", data["model_rows"])
        self._fill_table("#gpu-table", data["gpu_rows"])
        self.query_one("#status", Static).update(data["status"])

    def _fill_table(self, selector: str, rows: Iterable[tuple[str, ...]]) -> None:
        table = self.query_one(selector, DataTable)
        table.clear()
        for cells in rows:
            table.add_row(*cells)
