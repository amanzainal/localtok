"""Generic OpenAI-compatible provider adapter.

Many local servers (llama.cpp's server, vLLM, LM Studio, text-generation-webui,
LocalAI, ...) expose ``GET /v1/models``. That endpoint only lists model ids —
it carries no VRAM or tok/s info — so this adapter reports models as
``available`` with a name and nothing else. It's the lowest common denominator
that works against almost any server.

An optional API key is read from an env var (never hardcoded). The key is only
sent if present; purely-local servers usually need none.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import httpx

from localtok.models import ModelRow, Status
from localtok.providers.base import PollResult, Provider

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_KEY_ENV = "LOCALTOK_OPENAI_API_KEY"


def parse_models(payload: dict[str, Any], provider_name: str) -> list[ModelRow]:
    """Parse a ``/v1/models`` response into ModelRows.

    Accepts the canonical ``{"data": [{"id": ...}, ...]}`` shape and tolerates
    a bare list for servers that return one.
    """
    if isinstance(payload, list):
        entries = payload
    else:
        entries = payload.get("data", []) or []

    rows: list[ModelRow] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry
        else:
            name = entry.get("id") or entry.get("name") or "?"
        rows.append(
            ModelRow(
                name=name,
                provider=provider_name,
                status=Status.AVAILABLE,
            )
        )
    rows.sort(key=lambda r: r.name.lower())
    return rows


class OpenAICompatProvider(Provider):
    name = "openai"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 3.0,
        api_key_env: str = DEFAULT_KEY_ENV,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(base_url, timeout)
        # Prefer an explicitly-passed key (e.g. from tests), else read env.
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    async def poll(self, client: httpx.AsyncClient) -> PollResult:
        result = PollResult()
        try:
            resp = await client.get(
                f"{self.base_url}/v1/models",
                timeout=self.timeout,
                headers=self._headers(),
            )
            resp.raise_for_status()
            result.models = parse_models(resp.json(), self.name)
        except Exception as exc:  # noqa: BLE001 - surface, don't crash
            result.errors.append(f"openai /v1/models: {exc}")
        return result
