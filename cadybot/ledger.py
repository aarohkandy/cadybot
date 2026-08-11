"""A daily close of every event count, so "what did this look like a fortnight
ago" has an answer.

`snapshot.build` recomputes everything from the raw tables and throws the result
away, which means cadybot has never been able to compare itself against its own
past. Every metric it reports is a reading with no history behind it. This file
keeps one row per metric per UTC day and nothing else.

Two rules decide what is allowed in here, and both exist because of the same
failure.

**Only event counts.** `LEDGER_METRICS` is `snapshot.SCOREABLE_METRICS` minus
`scorecard.NON_COUNT_METRICS`, which leaves eight paths that count things that
happened. The excluded ones are shares, ages, stocks and bounded indices, and
the important member of that set is `activity.days_since_owner_posted`: it rises
by exactly 1.0 every day *because nobody posts*. Anything watching it for
movement fires forever on a dead server, having learned nothing. A metric that
moves because time passed is a clock, and a clock is not news.

**Only whole days.** Sampling hourly would be 24x the rows for a finer answer
nobody asked, and it would make the listener's own downtime legible as a signal:
if the process is asleep from Friday to Sunday, an hourly series shows a cliff
and a spike that a daily one simply does not contain. A missing day here is
absent, not zero, and `value_on` returns None for it.

Model-free, like scorecard.py and for the same reason: nothing here may import
advisor or llm, and nothing here asks a model anything.
"""

import datetime
from typing import Any, Dict, List, Optional, Tuple

from . import db, scorecard, snapshot

# The eight paths worth a daily reading. Derived rather than typed out so that a
# metric added to SCOREABLE_METRICS is either counted here or explicitly
# classified as non-count in scorecard.py — there is no third place to forget.
LEDGER_METRICS: Tuple[str, ...] = tuple(
    m for m in snapshot.SCOREABLE_METRICS if m not in scorecard.NON_COUNT_METRICS
)

# How much history to keep. Long enough that a seasonal comparison is possible,
# short enough that the table cannot grow with uptime forever.
LEDGER_DAYS = 180

# The window `drift` looks back over. Matches the fortnight a recommendation
# gets by default (config.RECOMMENDATION_HORIZON_DAYS), so "has anything moved
# since the last bet was opened" and "has anything drifted" ask the same span.
DRIFT_DAYS = 14

# What counts as a real move. Mirrors scorecard._threshold's shape — an absolute
# floor so a count of 1 going to 2 is not a 100% swing, and a proportional one so
# a busy server is not tripped by noise.
DRIFT_MIN_ABS = 3.0
DRIFT_MIN_REL = 0.25


def today() -> str:
    """The current UTC calendar day. The ledger's unit."""
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


def day_offset(days: int) -> str:
    d = datetime.datetime.now(datetime.timezone.utc).date()
    return (d - datetime.timedelta(days=days)).isoformat()


def record_day(guild_id: int, snap: Dict[str, Any]) -> int:
    """Close today's readings. Idempotent — safe to call every hour.

    INSERT OR REPLACE rather than INSERT OR IGNORE so that the last reading of a
    day wins. The alternative is that a 00:04 sample taken four minutes into the
    day becomes the whole day's record.
    """
    conn = db.connect()
    day = today()
    written = 0
    for metric in LEDGER_METRICS:
        value = snapshot.resolve_metric(snap, metric)
        if value is None:
            continue  # absent is not zero, and a missing row says so
        conn.execute(
            "INSERT OR REPLACE INTO ledger (guild_id, day, metric, value) "
            "VALUES (?, ?, ?, ?)",
            (guild_id, day, metric, float(value)),
        )
        written += 1
    return written


def value_on(guild_id: int, day: str, metric: str) -> Optional[float]:
    """One reading, or None if that day was never closed."""
    row = db.one(
        "SELECT value FROM ledger WHERE guild_id=? AND day=? AND metric=?",
        (guild_id, day, metric),
    )
    return row["value"] if row else None


def series(guild_id: int, metric: str, days: int = 30) -> List[Dict[str, Any]]:
    """The readings for one metric, oldest first. For `cadybot ledger`."""
    return db.query(
        "SELECT day, value FROM ledger WHERE guild_id=? AND metric=? AND day>=? "
        "ORDER BY day",
        (guild_id, metric, day_offset(days)),
    )


def days_recorded(guild_id: int) -> int:
    """How many distinct days this guild has closed. The warm-up counter."""
    return db.scalar(
        "SELECT COUNT(DISTINCT day) FROM ledger WHERE guild_id=?", (guild_id,)
    )


def drift(guild_id: int) -> Optional[Tuple[str, float, float]]:
    """The first metric that moved materially over DRIFT_DAYS.

    Returns (metric, before, after) or None. Compares two *stored* closes, never
    a stored close against a live snapshot: reading one end from the ledger and
    the other from `snapshot.build` would make the comparison sensitive to what
    time of day the pass happened to run.

    Deliberately returns one metric rather than all of them. Eight metrics on a
    server that just woke up will all move at once off the same handful of
    messages, and a caller that treats each as a separate finding turns one
    event into eight.
    """
    before_day = day_offset(DRIFT_DAYS)
    for metric in LEDGER_METRICS:
        before = value_on(guild_id, before_day, metric)
        after = value_on(guild_id, today(), metric)
        if before is None or after is None:
            continue
        if abs(after - before) >= max(DRIFT_MIN_ABS, abs(before) * DRIFT_MIN_REL):
            return (metric, before, after)
    return None


def prune(guild_id: int) -> int:
    """Drop closes older than LEDGER_DAYS."""
    return db.connect().execute(
        "DELETE FROM ledger WHERE guild_id=? AND day<?",
        (guild_id, day_offset(LEDGER_DAYS)),
    ).rowcount
