"""Privacy guard: no personal data may leak into the repo.

The committed checks are deliberately GENERIC -- hardcoding real hostnames,
usernames or identities in this file would itself be the leak we are trying to
prevent. Two layers run:

1. Structural patterns that should never appear in an open-source repo
   (absolute home paths with a real username, CGNAT/VPN-range IPs).
2. An optional, project-specific denylist that is kept OUT of version control.
   Drop a `tests/privacy_denylist.local.txt` (one lowercase token per line) or
   point ``$LOCALTOK_PRIVACY_DENYLIST`` at one; those tokens are loaded at test
   time but never committed. See ``tests/privacy_denylist.example.txt``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "dist", "build", ".ruff_cache"}
SCAN_SUFFIXES = {".py", ".md", ".toml", ".json", ".csv", ".txt", ".cfg", ".ini", ".example", ""}

# Absolute home paths are fine when they use an obvious placeholder username,
# but a real username pinned into source is a leak.
PLACEHOLDER_NAMES = {"user", "you", "username", "me", "name", "someone", "example", "youruser"}
HOME_PATH_RE = re.compile(r"/(?:home|Users)/([A-Za-z][A-Za-z0-9_.-]+)")
# Carrier-grade NAT / common VPN overlay range (e.g. Tailscale 100.64/10).
VPN_IP_RE = re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")


def _iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "test_privacy.py":
            continue  # avoid matching our own regex source
        if path.suffix.lower() in SCAN_SUFFIXES:
            yield path


def _local_denylist() -> list[str]:
    """Project-specific tokens, loaded from a gitignored file (never committed)."""
    sources: list[Path] = []
    env = os.environ.get("LOCALTOK_PRIVACY_DENYLIST")
    if env:
        sources.append(Path(env))
    sources.append(ROOT / "tests" / "privacy_denylist.local.txt")
    tokens: list[str] = []
    for src in sources:
        if src.is_file():
            for line in src.read_text(errors="ignore").splitlines():
                line = line.strip().lower()
                if line and not line.startswith("#"):
                    tokens.append(line)
    return tokens


def test_no_real_home_paths():
    offenders = []
    for path in _iter_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for m in HOME_PATH_RE.finditer(text):
            if m.group(1).lower() not in PLACEHOLDER_NAMES:
                offenders.append(f"{path.relative_to(ROOT)}: real home path '{m.group(0)}'")
    assert not offenders, "Real home paths found:\n" + "\n".join(offenders)


def test_no_vpn_range_ips():
    offenders = []
    for path in _iter_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for m in VPN_IP_RE.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: VPN-range IP '{m.group(0)}'")
    assert not offenders, "VPN-range IPs found:\n" + "\n".join(offenders)


def test_no_local_denylist_tokens():
    tokens = _local_denylist()
    if not tokens:
        import pytest

        pytest.skip("no local denylist configured (see tests/privacy_denylist.example.txt)")
    offenders = []
    for path in _iter_files():
        try:
            text = path.read_text(errors="ignore").lower()
        except OSError:
            continue
        for token in tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains '{token}'")
    assert not offenders, "Denylist tokens found:\n" + "\n".join(offenders)
