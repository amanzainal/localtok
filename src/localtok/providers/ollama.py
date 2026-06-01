"""Ollama provider adapter.

Uses two endpoints:

* ``GET /api/ps``   — models currently loaded into memory. Each entry carries
  ``size`` (total) and ``size_vram`` (the portion resident in VRAM), which lets
  us show a real VRAM column for loaded models.
* ``GET /api/tags`` — every model the server knows about (the local library).
  These are marked ``available`` unless ``/api/ps`` already reported them as
  ``loaded``.

Parsing is split into pure functions so tests can run against captured JSON
with no Ollama process anywhere in sight.
"""

from __future__ import annotations

from typing import Any

import httpx

from localtok.models import ModelRow, Status
from localtok.providers.base import PollResult, Provider

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


def _quant_detail(details: dict[str, Any] | None) -> str:
    if not details:
        return ""
    parts = []
    fam = details.get("family")
    if fam:
        parts.append(str(fam))
    quant = details.get("quantization_level")
    if quant:
        parts.append(str(quant))
    return " ".join(parts)


def parse_ps(payload: dict[str, Any], provider_name: str) -> list[ModelRow]:
    """Parse an ``/api/ps`` response into loaded ModelRows."""
    rows: list[ModelRow] = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model") or "?"
        size = entry.get("size")
        vram = entry.get("size_vram")
        rows.append(
            ModelRow(
                name=name,
                provider=provider_name,
                status=Status.LOADED,
                size_bytes=int(size) if isinstance(size, (int, float)) else None,
                vram_bytes=int(vram) if isinstance(vram, (int, float)) else None,
                detail=_quant_detail(entry.get("details")),
            )
        )
    return rows


def parse_tags(payload: dict[str, Any], provider_name: str) -> list[ModelRow]:
    """Parse an ``/api/tags`` response into available ModelRows."""
    rows: list[ModelRow] = []
    for entry in payload.get("models", []) or []:
        name = entry.get("name") or entry.get("model") or "?"
        size = entry.get("size")
        rows.append(
            ModelRow(
                name=name,
                provider=provider_name,
                status=Status.AVAILABLE,
                size_bytes=int(size) if isinstance(size, (int, float)) else None,
                detail=_quant_detail(entry.get("details")),
            )
        )
    return rows


def merge_rows(loaded: list[ModelRow], available: list[ModelRow]) -> list[ModelRow]:
    """Combine loaded + available rows, preferring loaded data per model name.

    A model that appears in ``/api/ps`` is loaded (with VRAM info); the same
    name in ``/api/tags`` is dropped to avoid a duplicate row. Loaded models
    sort first, then everything else alphabetically.
    """
    by_name: dict[str, ModelRow] = {}
    for row in available:
        by_name[row.name] = row
    for row in loaded:  # loaded wins
        by_name[row.name] = row

    def sort_key(r: ModelRow) -> tuple[int, str]:
        return (0 if r.status == Status.LOADED else 1, r.name.lower())

    return sorted(by_name.values(), key=sort_key)


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> None:
        super().__init__(base_url, timeout)

    async def poll(self, client: httpx.AsyncClient) -> PollResult:
        result = PollResult()
        loaded: list[ModelRow] = []
        available: list[ModelRow] = []

        try:
            resp = await client.get(f"{self.base_url}/api/ps", timeout=self.timeout)
            resp.raise_for_status()
            loaded = parse_ps(resp.json(), self.name)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash
            result.errors.append(f"ollama /api/ps: {exc}")

        try:
            resp = await client.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            resp.raise_for_status()
            available = parse_tags(resp.json(), self.name)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"ollama /api/tags: {exc}")

        result.models = merge_rows(loaded, available)
        return result
