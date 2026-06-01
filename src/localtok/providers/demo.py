"""Synthetic demo provider.

Produces obviously-fake data so ``localtok --demo`` runs anywhere — no LLM
server, no GPU required. It powers the screenshot in the README and gives the
test-suite a deterministic-shaped (but lightly animated) source.

Everything here is fabricated: model names ``demo-7b`` / ``demo-70b``, a fake
GPU ``node-a``, and a tok/s readout that drifts on a sine wave so the live
table visibly updates.
"""

from __future__ import annotations

import math
import time

import httpx

from localtok.models import GpuRow, ModelRow, Status
from localtok.providers.base import PollResult, Provider

GB = 1024 ** 3


class DemoProvider(Provider):
    name = "demo"

    def __init__(self, base_url: str = "demo://localhost", timeout: float = 3.0) -> None:
        super().__init__(base_url, timeout)
        self._start = time.monotonic()

    def snapshot(self, t: float | None = None) -> PollResult:
        """Build a synthetic PollResult for elapsed time ``t`` seconds.

        Pure and side-effect free so tests can pin ``t`` and assert exact
        values. ``poll`` just calls this with real elapsed time.
        """
        if t is None:
            t = time.monotonic() - self._start

        # tok/s drifts on offset sine waves so the two models differ visibly.
        tps_small = 92.0 + 18.0 * math.sin(t / 2.0)
        tps_large = 21.0 + 5.0 * math.sin(t / 3.0 + 1.0)

        models = [
            ModelRow(
                name="demo-7b",
                provider=self.name,
                status=Status.LOADED,
                size_bytes=4 * GB + 700 * (GB // 1000),
                vram_bytes=5 * GB,
                tokens_per_sec=round(tps_small, 1),
                detail="example q4_k_m",
            ),
            ModelRow(
                name="demo-70b",
                provider=self.name,
                status=Status.LOADED,
                size_bytes=40 * GB,
                vram_bytes=42 * GB,
                tokens_per_sec=round(tps_large, 1),
                detail="example q4_k_m",
            ),
            ModelRow(
                name="example/base-model-v1",
                provider=self.name,
                status=Status.AVAILABLE,
                size_bytes=int(2.3 * GB),
                detail="example fp16",
            ),
        ]

        used = (5 + 42) * GB
        gpus = [
            GpuRow(
                index=0,
                name="node-a",
                mem_used_bytes=used,
                mem_total_bytes=48 * GB,
                utilization_pct=round(60.0 + 30.0 * abs(math.sin(t / 2.0)), 1),
                temperature_c=round(58.0 + 8.0 * abs(math.sin(t / 4.0)), 1),
            )
        ]

        return PollResult(models=models, gpus=gpus, errors=[])

    async def poll(self, client: httpx.AsyncClient) -> PollResult:
        return self.snapshot()
