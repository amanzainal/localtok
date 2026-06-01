"""Aggregate one dashboard refresh from many providers.

The :class:`Collector` owns a shared ``httpx.AsyncClient`` and a list of
providers. Each refresh fans out :meth:`Provider.poll` concurrently, merges the
results into a single :class:`Snapshot`, and (unless in demo mode) layers GPU
rows from ``nvidia-smi`` on top. Provider failures become ``errors`` on the
snapshot instead of exceptions.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

import httpx

from localtok import gpu as gpu_mod
from localtok.models import Snapshot
from localtok.providers.base import PollResult, Provider


class Collector:
    def __init__(
        self,
        providers: Sequence[Provider],
        use_gpu: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.providers = list(providers)
        self.use_gpu = use_gpu
        self._client = client
        self._owns_client = client is None

    async def __aenter__(self) -> "Collector":
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def refresh(self) -> Snapshot:
        """Poll every provider concurrently and assemble a Snapshot."""
        assert self._client is not None, "use Collector as an async context manager"

        async def _safe_poll(p: Provider) -> PollResult:
            try:
                return await p.poll(self._client)
            except Exception as exc:  # noqa: BLE001 - never let one provider abort
                return PollResult(errors=[f"{p.name}: {exc}"])

        results = await asyncio.gather(*(_safe_poll(p) for p in self.providers))

        snap = Snapshot()
        for res in results:
            snap.models.extend(res.models)
            snap.gpus.extend(res.gpus)
            snap.errors.extend(res.errors)

        if self.use_gpu:
            # nvidia-smi is a blocking subprocess; keep the event loop free.
            gpu_rows = await asyncio.to_thread(gpu_mod.query_gpus)
            snap.gpus.extend(gpu_rows)

        return snap
