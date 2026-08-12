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

# "internal" runs the reporting passes inside the listener; "cron" leaves them
# to an external scheduler calling `python -m cadybot post`. Exactly one must be
# in charge, or the founder gets every brief twice.
SCHEDULER = (os.getenv("CADYBOT_SCHEDULER") or "internal").strip().lower()

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
def _keep_alive():
    """Ollama's keep_alive, as a number of seconds when it is one.

    Sent as the JSON string "0" this is parsed as a duration, fails, and Ollama
    silently applies its five-minute default — so the setting that is supposed
    to free several gigabytes does the opposite of nothing. A bare integer is
    unambiguous; anything else ("30m", "1h") passes through untouched.
    """
    raw = (os.getenv("CADYBOT_OLLAMA_KEEP_ALIVE") or "30m").strip()
    try:
        return int(raw)
    except ValueError:
        return raw


OLLAMA_KEEP_ALIVE = _keep_alive()

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


# --- the desk: thinking without being asked --------------------------------
#
# cadybot wakes on a timer, but it only *thinks* when something happened — see
# agenda.py. These numbers bound what happens when something does.

# How often the desk checks whether anything has happened. Cheap: the common
# outcome is a handful of indexed SELECTs that find nothing and return.
THINK_INTERVAL_HOURS = 6

# discord.ext.tasks runs an interval loop once immediately on start, so with
# Restart=always a crash loop would attempt a pass every ten seconds — on the
# tick most likely to find the backend still coming up.
THINK_START_DELAY_SECONDS = 60

# Hard ceiling on thoughts per guild per rolling day. Counted from journal rows
# written *before* the call, so a failure is charged like a success — the calls
# that time out are the expensive ones. Set to 0 to stop thinking entirely
# without touching code.
THINK_CALLS_PER_DAY = _int("CADYBOT_THINK_CALLS_PER_DAY", 2)

# Backends allowed to speak unprompted. Thinking happens on any backend and the
# journal fills either way; this only controls initiating.
#
# The default is the API model, because an unprompted message is the one place
# with no founder question to anchor against and llm.py calls the local path
# noticeably worse at judgment. But it is a setting rather than a rule: the
# local model is what actually runs here, and a desk that can never speak is not
# a desk. Set CADYBOT_THINK_SURFACE_BACKENDS=ollama,anthropic to let it talk on
# local inference — the evidence gates in thinking.py apply either way, and they
# are stricter for an unprompted message than for an answer that was asked for.
THINK_SURFACE_BACKENDS = tuple(
    b.strip().lower()
    for b in (os.getenv("CADYBOT_THINK_SURFACE_BACKENDS") or "anthropic").split(",")
    if b.strip()
)

# At most one volunteered thought a week, never within 20h of anything else
# cadybot said, and only in the afternoon UTC window the weekly brief already
# uses. The gap is measured against every kind of delivery, so the nightly, the
# weekly, the unanswered-question alert and the desk finally know about each
# other.
SURFACE_MAX_PER_WEEK = 1
SURFACE_MIN_GAP_HOURS = 20
SURFACE_WINDOW_UTC = (14, 20)

# A note cadybot leaves itself rides into a later brief. Both numbers exist so
# carried context cannot grow with uptime.
NOTE_TTL_DAYS = 60
NOTES_CARRIED = 3


def require_discord() -> None:
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_TOKEN is not set. Copy .env.example to .env and fill it in.")

# How much looking-up a reply may do before answering. Chat gets two rounds
# because the one genuinely two-hop question in the tool surface is
# channel_map -> channel_messages; /ask gets one because it is a verdict, not a
# conversation. The desk gets none: it speaks unprompted, and widening the set
# of citable figures widens what it volunteers.
INQUIRY_ROUNDS_CHAT = _int("CADYBOT_INQUIRY_ROUNDS", 2)
INQUIRY_BUDGET_CHAT = _int("CADYBOT_INQUIRY_BUDGET", 150)
