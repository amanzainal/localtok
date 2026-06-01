"""Provider adapters for local LLM servers."""

from localtok.providers.base import PollResult, Provider
from localtok.providers.demo import DemoProvider
from localtok.providers.ollama import OllamaProvider
from localtok.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "PollResult",
    "Provider",
    "DemoProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
]
