"""Smoke test that the Textual dashboard mounts and paints rows.

Uses Textual's headless test pilot — no real terminal required — so the full
app (compose, mount, refresh timer wiring, table fill) is exercised in CI.
"""

from __future__ import annotations

import pytest

from localtok.app import LocaltokApp
from localtok.collector import Collector
from localtok.providers.demo import DemoProvider


@pytest.mark.asyncio
async def test_app_mounts_and_fills_tables():
    app = LocaltokApp(Collector([DemoProvider()], use_gpu=False), interval=0.1)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        from textual.widgets import DataTable

        model_table = app.query_one("#model-table", DataTable)
        gpu_table = app.query_one("#gpu-table", DataTable)
        assert model_table.row_count == 3
        assert gpu_table.row_count == 1

        # 'r' triggers an explicit refresh and the table stays populated.
        await pilot.press("r")
        await pilot.pause()
        assert app.query_one("#model-table", DataTable).row_count == 3
