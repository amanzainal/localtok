"""Privacy guard: no personal data may leak into the repo.

This test is the safety net for open-sourcing. It scans every tracked source,
test, doc and config file for a denylist of personal/host strings. If anyone
ever pastes a real path or hostname into the project, CI goes red here.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Lowercased substrings that must never appear in committed files.
BANNED = [
    "example-host",
    "example-host",
    "example-host",
    "example-host",
    "example-tool",
    "example-wiki",
    "example-bot",
    "example-agent",
    "example-vault",
    "amanzainal",
    "user@example.com",
    "/home/user",
    "example-subject",
    "example-subject",
    "example-subject",
]

# Files/dirs we don't scan (binary, caches, vcs, this test itself).
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "dist", "build", ".ruff_cache"}
SCAN_SUFFIXES = {".py", ".md", ".toml", ".json", ".csv", ".txt", ".cfg", ".ini", ".example", ""}


def _iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "test_privacy.py":
            continue  # this file legitimately names the banned strings
        if path.suffix.lower() in SCAN_SUFFIXES:
            yield path


def test_no_banned_strings_in_repo():
    offenders = []
    for path in _iter_files():
        try:
            text = path.read_text(errors="ignore").lower()
        except OSError:
            continue
        for token in BANNED:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)} contains '{token}'")
    assert not offenders, "Banned personal strings found:\n" + "\n".join(offenders)


def test_no_tailscale_ips():
    import re

    pattern = re.compile(r"\b100\.(?:\d{1,3})\.(?:\d{1,3})\.(?:\d{1,3})\b")
    offenders = []
    for path in _iter_files():
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for m in pattern.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: {m.group(0)}")
    assert not offenders, "Tailscale-range IPs found:\n" + "\n".join(offenders)
