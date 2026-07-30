"""Environment-backed configuration. Nothing here talks to Discord or a model."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def _int(name: str, default: Optional[int] = None) -> Optional[int]:
    raw = (os.getenv(name) or "").strip()
    return int(raw) if raw else default


def _required_int(name: str) -> int:
    value = _int(name)
    if value is None:
        raise SystemExit("%s is not set. Copy .env.example to .env and fill it in." % name)
    return value


DISCORD_TOKEN = (os.getenv("DISCORD_TOKEN") or "").strip()
GUILD_ID = _int("GUILD_ID")
OWNER_ID = _int("OWNER_ID")

DB_PATH = ROOT / (os.getenv("CADYBOT_DB") or "cadybot.db")
CONTEXT_DIR = ROOT / "context"
PLAYBOOK_DIR = ROOT / "playbooks"

# The private channel cadybot creates and manages. It is the only channel in the
# server cadybot is permitted to write to, and its member list is yours to edit.
ROOM_NAME = (os.getenv("CADYBOT_ROOM") or "cadybot").strip().lower()

# "ollama" for local inference, "anthropic" for Claude.
BACKEND = (os.getenv("CADYBOT_BACKEND") or "ollama").strip().lower()

# Opus 5 defaults to `effort: high` and to thinking on, which is what we want
# here, so neither is passed explicitly.
MODEL = os.getenv("CADYBOT_MODEL") or "claude-opus-5"

OLLAMA_HOST = (os.getenv("CADYBOT_OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("CADYBOT_OLLAMA_MODEL") or "gemma4:e4b"
# Ollama's default context is small enough to silently truncate the system
# prompt. A truncated stage gate is worse than none at all.
OLLAMA_NUM_CTX = _int("CADYBOT_OLLAMA_NUM_CTX", 16384)
OLLAMA_TIMEOUT = _int("CADYBOT_OLLAMA_TIMEOUT", 600)

# A member who has not posted in this many days counts as gone quiet.
QUIET_DAYS = 14
# A member question with no reply after this many hours is flagged as unanswered.
UNANSWERED_HOURS = 3


def require_discord() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.")
    _required_int("GUILD_ID")
    _required_int("OWNER_ID")
