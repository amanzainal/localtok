"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def load_json_fixture(name: str) -> dict:
    return json.loads(load_fixture(name))


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
