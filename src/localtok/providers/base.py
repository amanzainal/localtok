"""The provider interface every adapter implements.

A provider is a thin async wrapper around a local LLM server's HTTP API. It
exposes exactly one coroutine, :meth:`Provider.poll`, which returns the rows it
discovered. Network/parse failures are caught and surfaced as ``errors`` on the
returned partial result rather than raised, so one dead server never takes down
the whole dashboard.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import httpx

from localtok.models import GpuRow, ModelRow


@dataclass
class PollResult:
    """What a single provider returns for one poll."""

    models: list[ModelRow] = field(default_factory=list)
    gpus: list[GpuRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class Provider(abc.ABC):
    """Base class for all provider adapters.

    Subclasses must implement :meth:`poll`. They should *parse* server JSON via
    the synchronous ``parse_*`` helpers (kept separate so tests can exercise
    parsing against fixtures with no live server).
    """

    #: Short label shown in the table's "provider" column.
    name: str = "base"

    def __init__(self, base_url: str, timeout: float = 3.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @abc.abstractmethod
    async def poll(self, client: httpx.AsyncClient) -> PollResult:
        """Fetch current state from the server and return parsed rows."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"{type(self).__name__}(name={self.name!r}, base_url={self.base_url!r})"
