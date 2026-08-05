"""The scheduled passes, as a state machine in code rather than in a prompt.

Fixed order, every time:

  1. build the snapshot
  2. grade every recommendation past its horizon, and commit
  3. decide whether there is anything worth saying
  4. at most one model call, to narrate what step 2 already decided
  5. deliver through notify

Step 2 committing before step 4 begins is the whole reason this file exists. An
ollama call takes 20-90 seconds and sometimes does not come back; if grading
ran after it, every run that died would leave its rows ungraded. Those runs are
not a random sample — they correlate with long prompts and busy machines — so
the ungraded set would drift toward the weeks that went badly, and the scorecard
would quietly flatter itself. Grading first costs nothing and removes the whole
class of problem.

Nothing here decides a verdict. That is scorecard.py, which no model touches.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional

from . import advisor, config, db, llm, notify, scorecard, snapshot

# Per-guild setting holding the metric values as of the last report cadybot
# actually posted — not the last time this ran. A drift of two messages a night
# should eventually be worth mentioning; measuring against last night's numbers
# would mean it never was.
LAST_REPORT_KEY = "loop_last_report"

# The handful of numbers a nightly pass watches for movement. Kept short on
# purpose: every metric added here is another way to justify interrupting a
# founder who has seven members and nothing new to hear.
WATCHED = (
    "activity.messages_7d",
    "activity.unique_posters_7d",
    "members.humans",
)


def _watched_values(snap: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return dict((path, snapshot.resolve_metric(snap, path)) for path in WATCHED)


def _movement(guild_id: int, snap: Dict[str, Any]) -> Optional[str]:
    """What moved since the last report, or None if nothing moved enough.

    config.SPEAK_THRESHOLD_MESSAGES is read here as "how large a move is worth
    interrupting the founder for". It was introduced with a narrower meaning —
    lifetime messages before a member counts as having found their voice — and
    is reused rather than duplicated so there is one number to argue about.
    """
    raw = db.get_setting(guild_id, LAST_REPORT_KEY)
    if not raw:
        return "first report for this server"
    try:
        previous = json.loads(raw)
    except ValueError:
        return "first report for this server"

    current = _watched_values(snap)
    for path, value in current.items():
        before = previous.get(path)
        if before is None or value is None:
            continue
        if abs(value - before) > config.SPEAK_THRESHOLD_MESSAGES:
            return "%s moved %+d since the last report" % (path, int(value - before))
    return None


def _remember(guild_id: int, snap: Dict[str, Any]) -> None:
    db.set_setting(guild_id, LAST_REPORT_KEY, json.dumps(_watched_values(snap)))


def _scorecard_only(
    verdicts: List[Dict[str, Any]], open_row: Optional[Dict[str, Any]]
) -> str:
    """A report with no model in it at all.

    A report that says almost nothing three weeks running is the correct report
    for a seven-member server. It has to look like a normal report rather than
    like a week cadybot skipped, or the founder learns to read silence as a bug.
    """
    lines: List[str] = []
    if verdicts:
        lines += advisor.render_scorecard(verdicts)
    if open_row:
        lines += [
            "**%s is still open** (day %.0f of %d): %s"
            % (
                open_row["ref"],
                open_row["age_days"],
                open_row["horizon_days"] or config.RECOMMENDATION_HORIZON_DAYS,
                open_row["action"],
            ),
            "No verdict yet, nothing new to say.",
            "",
        ]
    elif not verdicts:
        lines += ["No verdict due, nothing new to say.", ""]
    return "\n".join(lines).strip()


async def _deliver(client, guild_id: int, text: str) -> str:
    if client is not None:
        await notify.deliver(client, guild_id, text)
    return text


async def _offload(client, fn, *args):
    """Run the slow, model-bearing step off the event loop — but only there.

    db.connect() caches a single sqlite connection and sqlite refuses to serve
    it to a second thread, so every hop between threads is a hazard. The
    scheduled path hops anyway, because a model call is 20-90 seconds and the
    gateway heartbeat cannot wait that long; that is the hop listener.py already
    makes for the same reason. The CLI has no heartbeat to keep alive, so it
    stays on the thread that owns the connection.

    Steps 1 to 3 never hop at all. Grading is fast and deterministic, and it is
    the step that must not be lost.
    """
    if client is None:
        return fn(*args)
    return await asyncio.to_thread(fn, *args)


def _narrator(verdicts: List[Dict[str, Any]]) -> Optional[str]:
    """Which backend writes the sentence about a verdict it did not produce.

    A recommendation is narrated by the other backend where one is reachable, so
    the model reading the scorecard is not the model that wrote the row.
    """
    if not verdicts:
        return None
    return llm.alternate_if_usable()


async def nightly(client, guild_id: int) -> Optional[str]:
    """One nightly pass. Returns the delivered text, or None if it stayed quiet.

    `client` may be None, which runs everything except delivery — that is what
    `python -m cadybot loop` uses.
    """
    snap = snapshot.build(guild_id)
    verdicts = scorecard.score(guild_id, snap)   # committed before anything slow runs
    open_row = scorecard.open_row(guild_id)
    moved = _movement(guild_id, snap)

    if not verdicts and not moved:
        return None  # nothing due, nothing moved: no model call, no message

    if open_row or not moved:
        # A verdict landed, or the slot is taken. Either way there is no new
        # advice to write, so there is nothing for a model to do.
        _remember(guild_id, snap)
        return await _deliver(client, guild_id, _scorecard_only(verdicts, open_row))

    result = await _offload(
        client, advisor.brief, snap, guild_id, verdicts, _narrator(verdicts)
    )
    _remember(guild_id, snap)
    return await _deliver(
        client, guild_id, "**Nightly**\n\n" + advisor.render_brief(result)
    )


async def weekly(client, guild_id: int) -> Optional[str]:
    """One weekly pass. Always reports; only sometimes opens a recommendation."""
    snap = snapshot.build(guild_id)
    verdicts = scorecard.score(guild_id, snap)   # committed before anything slow runs
    open_row = scorecard.open_row(guild_id)

    if open_row:
        # One open bet per guild. With seven members and a fortnight, two live
        # interventions cannot be separated even in principle, so a second one
        # would only give the next grading pass something to attribute freely.
        _remember(guild_id, snap)
        return await _deliver(
            client,
            guild_id,
            "**Weekly brief**\n\n" + _scorecard_only(verdicts, open_row),
        )

    result = await _offload(
        client, advisor.brief, snap, guild_id, verdicts, _narrator(verdicts)
    )
    _remember(guild_id, snap)
    return await _deliver(
        client, guild_id, "**Weekly brief**\n\n" + advisor.render_brief(result)
    )
