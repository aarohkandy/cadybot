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

# Both optional. cadybot works in every server it is added to; each server gets
# its own private channel, its own owner, and its own isolated data. These two
# only set a default for the CLI so you don't have to pass --guild every time.
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
# How long Ollama keeps the model in memory after a request. The default is 5
# minutes, which means most questions pay a multi-GB reload before answering.
OLLAMA_KEEP_ALIVE = os.getenv("CADYBOT_OLLAMA_KEEP_ALIVE") or "30m"

# A member who has not posted in this many days counts as gone quiet.
QUIET_DAYS = 14
# A member question with no reply after this many hours is flagged as unanswered.
UNANSWERED_HOURS = 3

# How long after a question a later message may still count as its answer. Wider
# than UNANSWERED_HOURS on purpose: the alert should fire while the founder can
# still fix it, but a reply the next morning did answer the question.
RESPONSE_WINDOW_HOURS = 48
assert RESPONSE_WINDOW_HOURS >= UNANSWERED_HOURS, (
    "RESPONSE_WINDOW_HOURS must not be shorter than UNANSWERED_HOURS, or a "
    "question could be alerted on and then never creditable as answered."
)

# Floors below which a derived statistic is not reported at all. A 7-member
# server produces ratios that look precise and mean nothing; refusing to print
# them is the only honest option, and it is cheaper to agree on the floors here
# than to relitigate them at every call site.
MIN_RATE_DENOMINATOR = 20        # events needed before a percentage is shown
MIN_SHARE_DENOMINATOR = 30       # messages needed before a share-of-total is shown
MIN_RECIPROCITY_POSTERS = 15     # distinct posters needed for a reply graph
MIN_RECIPROCITY_EDGES = 30       # reply pairs needed for a reply graph
MIN_COHORT = 20                  # members needed before a cohort is compared
MIN_VERDICT_EVENTS = 20          # events in a baseline before a verdict is issued

# Messages closer together than this are one conversation, not two.
BURST_MINUTES = 5
# How often the online/member estimate is sampled from Discord.
PRESENCE_SAMPLE_HOURS = 1
# How far ahead a recommendation is allowed to promise anything.
RECOMMENDATION_HORIZON_DAYS = 14
# Lifetime messages after which a member counts as having found their voice.
SPEAK_THRESHOLD_MESSAGES = 10


def require_discord() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.")
