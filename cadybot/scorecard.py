"""Grading of pre-registered recommendations. Deterministic, no model.

snapshot.py is what stops cadybot hallucinating a member count. This module is
what stops it hallucinating that its advice helped. Same rule, applied to
judgments instead of statistics: every value below is computed from the database
and from the snapshot, and nothing here may import advisor or llm or ask a model
anything. A verdict arrives at the prompt as a fact the model did not produce
and cannot revise.

A verdict of `worked` means all four of these, and it is refused if any one of
them is missing:

  - the owner's own behaviour changed after the row was issued, measured against
    what they were already doing;
  - the metric named *before* the fact moved in the direction that is *better*
    for the server, past a threshold chosen *before* the fact;
  - that metric is a count of events, read twice over windows far enough apart
    that the two readings are not the same events;
  - the baseline was large enough that the move is separable from the noise in a
    count of that size.

It does not mean the advice caused the movement. One server, no control, no
randomisation — attribution is not available, and the word stays narrow on
purpose.

The bar is high enough that a seven-member server will mostly see
`inconclusive`, and that is the honest reading: at four messages a week nothing
that happens is distinguishable from nothing happening. A scorer that says so is
worth more than one that hands out a win every fortnight.

Do not add a self-critique or "reconsider the verdict" pass, here or upstream.
Huang et al. (ICLR 2024) measured GPT-3.5 flipping 7.6% of its wrong GSM8K
answers right and 8.8% of its right ones wrong when asked to reconsider without
new information — a net loss. Every reflective step must be anchored to a number
the model did not produce, which is what this file exists to supply.

That rule was once absolute. Since agenda.py it is exact rather than absolute,
and the difference is worth stating precisely: **no reflective step without a
fact that did not exist when the claim was made, where "fact" means a row in the
database.** agenda.next_provocation enforces it mechanically — a provocation's
timestamp is read from stored data and asserted to be in the past, so a pass
cannot reconsider anything merely because time went by. The reflection a closed
verdict provokes is handed the verdict this file computed, in a schema with no
field to revise it. That is Huang's oracle condition, not its absence.

What remains forbidden, and is not weakened: nothing may re-grade a row, no
model output may become a verdict, and no verdict may be revisited because a
model thought better of it. `_recheck` reopens a `worked` row only on a later
reading of the same pre-registered metric.
"""

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config, db, snapshot, stats

# The whole vocabulary. Every closed row holds exactly one of these.
VERDICTS: Tuple[str, ...] = (
    "worked",         # enacted, and the pre-registered metric cleared its threshold
    "failed",         # enacted, and it did not
    "harmful",        # a guardrail metric went through its floor, testably
    "revoked",        # graded worked, and then the metric did not stay there
    "inconclusive",   # the move is not distinguishable from noise
    "not_attempted",  # no positive evidence the founder did the thing
    "unmeasurable",   # the row names nothing this scorer is able to grade
)

OPEN_VERDICT = None  # a row with verdict IS NULL is still inside its horizon

# Fraction of baseline a metric must move before the move counts as the
# predicted one. Fixed in code, applied at issue time, and never chosen by the
# advisor: a threshold picked after the fact is picked to be cleared.
MIN_EFFECT = 0.25

# Two-sided alpha for the count test. Below this the delta is reported as
# inconclusive whichever way it points.
ALPHA = 0.05

# stats.count_change_pvalue builds the exact conditional binomial pmf, which
# multiplies math.comb(k, i) by a float. Above roughly a thousand total events
# that integer no longer converts to a float and the call raises. At that size
# the normal approximation to the binomial is excellent anyway, so the exact
# test is used where it matters (small counts) and approximated where it does
# not.
EXACT_MAX_EVENTS = 1000

# Which direction is *better* for each scoreable metric. A guardrail needs to
# know what "worse" means, and the answer cannot come from the model — asked
# after the fact which way a metric was supposed to go, it will say whichever
# way it went. +1 means higher is better, -1 means lower is better.
POLARITY: Dict[str, int] = {
    "activity.messages_7d": 1,
    "activity.messages_30d": 1,
    "activity.unique_posters_7d": 1,
    "activity.days_since_owner_posted": -1,
    "members.humans": 1,
    "members.never_posted.count": -1,
    "members.gone_quiet.count": -1,
    "response_rate.answered": 1,
    "response_rate.asked": 1,
    "communicators_30d": 1,
    "bus_factor_30d.factor": 1,
    "bus_factor_30d.contributors": 1,
    "top_share_30d.top1.share": -1,
    "top_share_30d.owner.share": -1,
    "membership_flow_30d.joins": 1,
    "membership_flow_30d.leaves": -1,
    "voice_30d.unique_participants": 1,
    "threads.opened_30d": 1,
    "structure.mentions_30d": 1,
    "retention_bracket.count": 1,
    "lurker_conversion.count": 1,
    "dead_channels.count": -1,
}

# The conditional binomial tests two counts of events accumulated over a fixed
# exposure, and nothing else. Everything listed here is something else:
#
#   - a share, or an age: there is no exposure to condition on;
#   - a stock — how many members or channels are in some state right now. That
#     is a state of the world rather than a count of events, and the two
#     readings are mostly the same people;
#   - a bounded index: the bus factor is an integer in [1, contributors];
#   - a distinct-actor cardinality, capped by the member count, so its variance
#     is nothing like its mean at any server size;
#   - a cohort numerator, whose denominator is a different set of people each
#     time it is read.
#
# Rows naming one of these resolve inconclusive rather than borrowing a test
# that does not apply to them.
NON_COUNT_METRICS = frozenset(
    ["top_share_30d.top1.share", "top_share_30d.owner.share",
     "activity.days_since_owner_posted",
     "members.humans", "members.never_posted.count", "members.gone_quiet.count",
     "dead_channels.count",
     "bus_factor_30d.factor", "bus_factor_30d.contributors",
     "activity.unique_posters_7d", "communicators_30d",
     "voice_30d.unique_participants",
     "retention_bracket.count", "lurker_conversion.count"]
)

# How many days of history stand behind one reading of each metric. Two readings
# taken less than this far apart are partly the same events counted twice:
# activity.messages_30d re-read a week later is 77% the same messages, and a
# test that treats those as two independent samples reads the shared data as
# agreement. The horizon is raised to the window at issue time, and grading
# refuses the comparison if the readings are closer than this anyway.
METRIC_WINDOW_DAYS: Dict[str, int] = {
    "activity.messages_7d": 7,
    "activity.messages_30d": 30,
    "activity.unique_posters_7d": 7,
    "members.gone_quiet.count": config.QUIET_DAYS,
    "response_rate.answered": 30,
    "response_rate.asked": 30,
    "communicators_30d": 30,
    "bus_factor_30d.factor": 30,
    "bus_factor_30d.contributors": 30,
    "top_share_30d.top1.share": 30,
    "top_share_30d.owner.share": 30,
    "membership_flow_30d.joins": 30,
    "membership_flow_30d.leaves": 30,
    "voice_30d.unique_participants": 30,
    "threads.opened_30d": 30,
    "structure.mentions_30d": 30,
    "retention_bracket.count": 14,
    "lurker_conversion.count": 30,
    "dead_channels.count": 30,
}

# Snapshot paths recorded at issue time so enactment can be established later.
# The first two are the ones _enactment grades; the rest are recorded because a
# baseline cannot be added to a row retroactively, and because they are what the
# CLI shows when somebody asks what the server looked like when a bet was made.
ENACTMENT_PATHS: Tuple[str, ...] = (
    "activity.owner_messages_7d",
    "channels.count",
    "activity.days_since_owner_posted",
    "response_rate.answered",
    "dead_channels.count",
    "threads.opened_30d",
)

# Columns the pre-registration needs that the original recommendations table did
# not have. Added here rather than in db.MIGRATIONS because this module owns
# them; the PRAGMA guard makes it a no-op if they later arrive from there too.
_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("direction", "TEXT"),
    ("horizon_days", "INTEGER"),
    ("guardrail_metric", "TEXT"),
    ("baseline", "REAL"),
    ("threshold", "REAL"),
    ("guardrail_baseline", "REAL"),
    ("guardrail_floor", "REAL"),
    ("baseline_json", "TEXT"),
    ("issued_by", "TEXT"),
    ("verdict", "TEXT"),
    ("verdict_at", "TEXT"),
    ("verdict_pvalue", "REAL"),
    ("verdict_current", "REAL"),
    ("verdict_source", "TEXT"),
    ("verdict_revoked_at", "TEXT"),
    ("enacted", "INTEGER"),
    ("enactment_evidence", "TEXT"),
)

_ensured_for: Optional[str] = None


def ensure_schema() -> None:
    """Add the pre-registration columns if this database predates them.

    Guarded by the PRAGMA rather than by catching the error, the same way
    db._migrate is, so a second run is a silent no-op.
    """
    global _ensured_for
    path = str(config.DB_PATH)
    if _ensured_for == path:
        return
    conn = db.connect()
    present = {row[1] for row in conn.execute("PRAGMA table_info(recommendations)")}
    if present:
        for column, decl in _COLUMNS:
            if column not in present:
                conn.execute("ALTER TABLE recommendations ADD COLUMN %s %s" % (column, decl))
    _ensured_for = path


# --- small helpers ---------------------------------------------------------


def ref(row_id: int) -> str:
    """How a recommendation is named everywhere it is shown or narrated.

    Never "my recommendation" and never "your advisor said". Panickssery et al.
    (NeurIPS 2024) found self-preference tracks self-recognition; recognisable
    authorship is one of the amplifiers, and it is free to remove.
    """
    return "R-%d" % row_id


def _parse(iso_ts: Optional[str]) -> Optional[datetime]:
    if not iso_ts:
        return None
    try:
        parsed = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _age_days(iso_ts: Optional[str]) -> Optional[float]:
    when = _parse(iso_ts)
    if when is None:
        return None
    return round((datetime.now(timezone.utc) - when).total_seconds() / 86400.0, 1)


def _horizon(row: Any) -> int:
    return int(row["horizon_days"] or config.RECOMMENDATION_HORIZON_DAYS)


def window_days(metric: str) -> int:
    """Days of history behind one reading of `metric`; 0 if it is instantaneous."""
    return METRIC_WINDOW_DAYS.get(metric, 0)


def _threshold(baseline: float, direction: str) -> float:
    """The bar, fixed at issue time from the baseline and the predicted sign.

    A metric already at zero cannot fall, so a "down" prediction against a zero
    baseline gets an unreachable threshold. That is the honest result: the
    prediction had no way to come true.
    """
    if direction == "up":
        return max(baseline + 1.0, baseline * (1.0 + MIN_EFFECT))
    if direction == "down":
        return min(baseline - 1.0, baseline * (1.0 - MIN_EFFECT))
    return baseline


def _cleared(current: float, baseline: float, threshold: float, direction: str) -> bool:
    if direction == "up":
        return current >= threshold
    if direction == "down":
        return current <= threshold
    return abs(current - baseline) <= max(1.0, abs(baseline) * MIN_EFFECT)


def _floor(baseline: float, polarity: int) -> float:
    """The worst a guardrail metric may get before the row is called harmful.

    For a lower-is-better metric this is an upper bound; the column is still
    called a floor, because what it bounds is how bad things may get.
    """
    if polarity > 0:
        return min(baseline - 1.0, baseline * (1.0 - MIN_EFFECT))
    return max(baseline + 1.0, baseline * (1.0 + MIN_EFFECT))


def _breached(current: float, floor: float, polarity: int) -> bool:
    return current < floor if polarity > 0 else current > floor


def _pvalue(baseline: float, current: float) -> float:
    """Two-sided p for a change between two counts over equal, disjoint windows.

    p0 is 0.5 because baseline and current are the same metric over the same
    window length; the exposures are equal by construction. Equal is not enough
    — they also have to be disjoint, or the shared events appear in both counts
    and the test reads them as agreement. Callers must not reach this until the
    two readings are at least window_days(metric) apart; _grade enforces that,
    and stats.count_change_pvalue's own docstring is the reason it matters.
    """
    x1, x2 = int(round(baseline)), int(round(current))
    if x1 < 0 or x2 < 0:
        return 1.0
    k = x1 + x2
    if k <= 0:
        return 1.0
    if k <= EXACT_MAX_EVENTS:
        return stats.count_change_pvalue(x1, x2, 0.5)
    z = (x1 - k * 0.5) / math.sqrt(k * 0.25)
    return math.erfc(abs(z) / math.sqrt(2.0))


def _snapshot_values(snap: Dict[str, Any], paths: Tuple[str, ...]) -> Dict[str, Optional[float]]:
    return dict((path, snapshot.resolve_metric(snap, path)) for path in paths)


def capped(rows: List[Dict[str, Any]], limit: int = 4) -> Dict[str, Any]:
    """{count, sample, note}, the same shape snapshot.py uses for its lists.

    The scorecard is one more list that grows with time, and a prompt block that
    grows without bound is the failure snapshot.py already had once.
    """
    out: Dict[str, Any] = {"count": len(rows)}
    if len(rows) <= limit:
        out["all"] = rows
    else:
        out["sample"] = rows[:limit]
        out["note"] = (
            "this is a sample of %d; the real total is %d — do not count the sample"
            % (limit, len(rows))
        )
    return out


# --- pre-registration ------------------------------------------------------


def prediction_text(metric: str, direction: str, threshold: Optional[float], horizon: int) -> str:
    """The commitment, in one line, composed by code rather than by the model.

    The stored sentence has to be immutable to be a pre-registration at all, so
    nothing that writes it may be prose the narrator can later reinterpret.
    """
    if metric == "none":
        return "no snapshot metric named; not gradable"
    if threshold is None:
        # The metric was absent from the snapshot when the bet was made, so no
        # bar exists. "past 0" is what `threshold or 0.0` used to print, and it
        # is a promise about a number nobody has: 0 is a bar every count clears.
        return (
            "%s was not in the snapshot at issue time, so no threshold could be "
            "fixed; not gradable" % metric
        )
    if direction == "unchanged":
        return "%s holds within %d%% of %.4g for %d days" % (
            metric, int(MIN_EFFECT * 100), threshold, horizon
        )
    return "%s goes %s, past %.4g, within %d days" % (metric, direction, threshold, horizon)


def pre_register(
    guild_id: int,
    snap: Dict[str, Any],
    recs: List[Dict[str, Any]],
    source: Optional[str] = None,
) -> List[int]:
    """Store recommendations with everything needed to grade them later.

    Baseline, threshold and guardrail floor are all read or derived here, at
    issue time. Nothing about the bar can be renegotiated once the horizon is
    running.
    """
    ensure_schema()
    stamp = db.now()
    baselines = json.dumps(_snapshot_values(snap, ENACTMENT_PATHS))
    ids: List[int] = []
    conn = db.connect()
    for rec in recs:
        metric = rec.get("metric") or "none"
        direction = rec.get("direction") or "unchanged"
        horizon = int(rec.get("horizon_days") or config.RECOMMENDATION_HORIZON_DAYS)
        # A metric that looks 30 days back cannot be judged in 7: the reading at
        # day 7 is three quarters the same events as the baseline, so the two
        # are not two samples of anything. The horizon is raised to the window
        # here, by code, and the stored prediction quotes the raised number —
        # the founder is told the date they are actually held to.
        horizon = max(horizon, window_days(metric))
        guardrail = rec.get("guardrail_metric") or "none"

        baseline = snapshot.resolve_metric(snap, metric) if metric != "none" else None
        threshold = _threshold(baseline, direction) if baseline is not None else None
        guard_baseline = (
            snapshot.resolve_metric(snap, guardrail) if guardrail != "none" else None
        )
        guard_floor = (
            _floor(guard_baseline, POLARITY.get(guardrail, 1))
            if guard_baseline is not None
            else None
        )

        cursor = conn.execute(
            "INSERT INTO recommendations "
            "(guild_id, created_at, headline, action, evidence, metric, prediction, "
            " direction, horizon_days, guardrail_metric, baseline, threshold, "
            " guardrail_baseline, guardrail_floor, baseline_json, issued_by) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                guild_id,
                stamp,
                rec.get("headline", ""),
                rec.get("action", ""),
                rec.get("evidence", ""),
                metric,
                prediction_text(metric, direction, threshold, horizon),
                direction,
                horizon,
                guardrail,
                baseline,
                threshold,
                guard_baseline,
                guard_floor,
                baselines,
                source,
            ),
        )
        ids.append(int(cursor.lastrowid))
    return ids


# --- reads -----------------------------------------------------------------


def open_row(guild_id: int) -> Optional[Dict[str, Any]]:
    """The one recommendation still inside its horizon, if there is one.

    With seven members and a fortnight to work with, two live interventions
    cannot be told apart even in principle. The model asked to attribute a move
    between them will attribute it generously, so only one is ever open.
    """
    ensure_schema()
    for row in db.query(
        "SELECT * FROM recommendations WHERE guild_id=? AND verdict IS NULL "
        "ORDER BY created_at DESC",
        (guild_id,),
    ):
        age = _age_days(row["created_at"])
        if age is not None and age < _horizon(row):
            out = dict(row)
            out["ref"] = ref(row["id"])
            out["age_days"] = age
            out["days_left"] = round(_horizon(row) - age, 1)
            return out
    return None


def _due(guild_id: int) -> List[Any]:
    """Open rows past their horizon, plus any row whose age cannot be computed.

    A row with an unparseable created_at has no horizon, so it can never come
    due. Left out it stays open forever — invisible, and holding the one open
    slot against every future recommendation. It is collected here so grading
    closes it as unmeasurable and somebody can see it.
    """
    ensure_schema()
    due: List[Any] = []
    for row in db.query(
        "SELECT * FROM recommendations WHERE guild_id=? AND verdict IS NULL "
        "ORDER BY created_at",
        (guild_id,),
    ):
        age = _age_days(row["created_at"])
        if age is None or age >= _horizon(row):
            due.append(row)
    return due


def recent_verdicts(guild_id: int, limit: int = 8) -> List[Dict[str, Any]]:
    """Closed rows, newest first. Reads only — grading happens in `score`."""
    ensure_schema()
    rows = db.query(
        "SELECT * FROM recommendations WHERE guild_id=? AND verdict IS NOT NULL "
        "ORDER BY COALESCE(verdict_at, created_at) DESC LIMIT ?",
        (guild_id, limit),
    )
    return [_as_report(row) for row in rows]


def _as_report(row: Any) -> Dict[str, Any]:
    """One closed row, as the numbers a report or a prompt is allowed to show."""
    baseline = row["baseline"]
    current = row["verdict_current"]
    delta = None
    if baseline is not None and current is not None:
        delta = round(current - baseline, 3)
    return {
        "ref": ref(row["id"]),
        "action_text": row["action"],          # verbatim; never paraphrased
        "headline": row["headline"],
        "created_at": row["created_at"],
        "metric": row["metric"],
        "direction": row["direction"],
        "horizon_days": row["horizon_days"],
        "prediction": row["prediction"],
        "baseline": baseline,
        "current": current,
        "delta": delta,
        "p_value": row["verdict_pvalue"],
        "verdict": row["verdict"],
        "verdict_at": row["verdict_at"],
        "enacted": row["enacted"],
        "enactment_evidence": row["enactment_evidence"],
        "guardrail_metric": row["guardrail_metric"],
        "revoked_at": row["verdict_revoked_at"],
    }


# --- grading ---------------------------------------------------------------


def _enactment(snap: Dict[str, Any], row: Any) -> Optional[str]:
    """Positive evidence that the founder did something. None means no evidence.

    What counts is a *change* in what the owner does, measured against what the
    owner was already doing when the row was issued. "The owner has posted
    recently" is true of an active founder on the day the recommendation is
    written and on every day after it, so it separates nothing: it graded every
    open row as enacted for anyone who uses their own server at all.

    Two signals qualify, both recorded in baseline_json at issue time:

      - activity.owner_messages_7d, the only thing in the snapshot attributable
        to the owner, held to the same bar the pre-registered metric has to
        clear;
      - channels.count, which moves only when somebody holding Manage Channels
        deliberately changes the server's shape. cadybot cannot see who that
        was, and the sentence below does not claim it was the owner.

    Replies and threads are deliberately not here. snapshot._ANSWERED counts an
    answer from any author other than the asker, and threads.opened_30d counts a
    thread opened by anyone, so both rise while the owner is silent for 45 days.
    Reading either as "the founder acted" is a fabricated attribution, emitted
    by the one layer whose whole purpose is to be unfabricatable.

    The gate is coarse — cadybot cannot read a free-text action and go check it
    — and it is biased toward not_attempted, which is the safe direction: it
    costs a true positive rather than manufacturing one.
    """
    try:
        before = json.loads(row["baseline_json"] or "{}")
    except (TypeError, ValueError):
        before = {}
    now = _snapshot_values(snap, ENACTMENT_PATHS)

    owner_before = before.get("activity.owner_messages_7d")
    owner_now = now.get("activity.owner_messages_7d")
    if (
        owner_before is not None
        and owner_now is not None
        and owner_now >= _threshold(owner_before, "up")
    ):
        return (
            "the owner posted %d message(s) in the last 7 days, against %d in the "
            "7 days before %s was issued"
            % (int(owner_now), int(owner_before), ref(row["id"]))
        )

    channels_before = before.get("channels.count")
    channels_now = now.get("channels.count")
    if channels_before is not None and channels_now is not None and channels_now != channels_before:
        return (
            "the channel list changed: %d channels, was %d at issue time — cadybot "
            "cannot see who changed it" % (int(channels_now), int(channels_before))
        )

    return None


def _wrong_way(metric: str, direction: str) -> bool:
    """Whether the pre-registered direction is the one that makes the server worse.

    POLARITY was consulted only by the guardrail, so nothing stopped a row
    naming members.gone_quiet.count and direction "up": it reads like a
    confident, measurable, pre-registered bet, and it graded `worked` when
    eighteen more people went silent. advisor.py refuses the pair before it can
    be stored; this is the same rule applied to rows stored before it existed.
    """
    polarity = POLARITY.get(metric)
    if polarity is None or direction == "unchanged":
        return False
    return (direction == "up") != (polarity > 0)


def _equivalence(baseline: float, current: float, pvalue: float) -> Tuple[str, str]:
    """Grade a `direction: unchanged` row, which claims the metric holds.

    Under the difference test alone this direction could not be graded at all:
    a metric that held gave p > ALPHA and resolved inconclusive, and one that
    moved failed _cleared — no input succeeded, while prediction_text went on
    promising the founder a verdict.

    Holding is a claim in its own right, and it needs the opposite reasoning to
    a difference: "no change was detected" is worthless when nothing could have
    been detected, so the count also has to be large enough that a move of the
    full band would have shown. That is the equivalence half, and it is why an
    `unchanged` row on a thin week is inconclusive rather than a free win.
    """
    if pvalue <= ALPHA:
        return ("failed", "the metric changed (p=%.3f); it was pre-registered to hold" % pvalue)
    if not _cleared(current, baseline, baseline, "unchanged"):
        return (
            "inconclusive",
            "the metric left its %d%% band but p=%.3f, so the move cannot be told "
            "from noise either" % (int(MIN_EFFECT * 100), pvalue),
        )
    if _pvalue(baseline, _threshold(baseline, "up")) > ALPHA:
        return (
            "inconclusive",
            "the metric held, but at a baseline of %d a move of the whole %d%% band "
            "would not have been detectable either — holding is not evidence here"
            % (int(round(baseline)), int(MIN_EFFECT * 100)),
        )
    return (
        "worked",
        "the metric held inside its %d%% band, at a count where a move of that size "
        "would have been separable from noise" % int(MIN_EFFECT * 100),
    )


def _grade(row: Any, snap: Dict[str, Any]) -> Dict[str, Any]:
    """One row, one verdict. Precedence is fixed and is the whole design."""
    metric = row["metric"] or "none"
    baseline = row["baseline"]
    threshold = row["threshold"]
    direction = row["direction"] or "unchanged"
    current = snapshot.resolve_metric(snap, metric) if metric != "none" else None
    age = _age_days(row["created_at"])
    window = window_days(metric)

    evidence = _enactment(snap, row)
    enacted = 1 if evidence else 0
    pvalue: Optional[float] = None
    note = ""
    breached, guard_note = _guardrail(row, snap)

    if age is None:
        verdict = "unmeasurable"
        note = "created_at %r cannot be read as a date, so this row has no horizon" % (
            row["created_at"],
        )
    elif metric == "none" or metric not in snapshot.SCOREABLE_METRICS:
        verdict = "unmeasurable"
        note = "no snapshot metric was pre-registered"
    elif not enacted:
        # The enactment gate. `worked` needs the founder to have done the thing
        # AND the number to have moved; no evidence of the first makes the
        # second uninterpretable, however it went.
        verdict = "not_attempted"
        note = "no evidence in the snapshot that the action was taken"
    elif breached:
        # Outranks the primary metric clearing its threshold on purpose: an
        # intervention that moved its target and broke something else is not a
        # success with a caveat.
        verdict = "harmful"
        note = guard_note or "guardrail %s went through its floor" % row["guardrail_metric"]
    elif current is None or baseline is None:
        verdict = "inconclusive"
        note = "the metric is not present in the current snapshot"
    elif _wrong_way(metric, direction):
        verdict = "unmeasurable"
        note = (
            "%s going %s is the direction that makes the server worse, so this row "
            "could never have been graded a success" % (metric, direction)
        )
    elif metric in NON_COUNT_METRICS:
        verdict = "inconclusive"
        note = (
            "%s is a stock, a share or a bounded index — the count test does not "
            "apply to it" % metric
        )
    elif age < window:
        verdict = "inconclusive"
        note = (
            "the two readings of %s are %.1f days apart against a %d-day window, so "
            "they are largely the same events counted twice" % (metric, age, window)
        )
    elif not stats.gradeable_count(int(round(baseline))):
        verdict = "inconclusive"
        note = (
            "the pre-registered baseline is %d, under the %d events a verdict needs; "
            "a count this small moves on its own" % (int(round(baseline)), stats.MIN_VERDICT_EVENTS)
        )
    else:
        pvalue = _pvalue(baseline, current)
        if direction == "unchanged":
            verdict, note = _equivalence(baseline, current, pvalue)
        elif pvalue > ALPHA:
            # Regardless of which way the delta points. A move in the predicted
            # direction that the test cannot separate from noise is noise.
            verdict = "inconclusive"
            note = "p=%.3f — not separable from noise at this count" % pvalue
        elif _cleared(current, baseline, threshold if threshold is not None else baseline, direction):
            verdict = "worked"
            note = "metric moved as pre-registered; this records movement, not causation"
        else:
            verdict = "failed"
            note = "metric did not clear its pre-registered threshold"

    if guard_note and verdict != "harmful":
        # The guardrail moved but could not be tested. Saying so is not a
        # verdict, and it is the difference between a founder hearing about it
        # and the number vanishing because it failed a significance test.
        note = "%s; %s" % (note, guard_note)

    _write_verdict(row["id"], verdict, pvalue, current, enacted, evidence)

    report = _as_report(
        db.one("SELECT * FROM recommendations WHERE id=?", (row["id"],))
    )
    report["note"] = note
    report["newly_closed"] = True
    return report


def _guardrail(row: Any, snap: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """(breached, note). A breach has to clear the bar a success has to clear.

    The primary metric needs a pre-registered threshold and p <= ALPHA before it
    may be called `worked`. A guardrail tested against its floor alone needs
    neither, and `harmful` outranks everything, so two of seven members going
    quiet — a normal fortnight — closed a row as harmful while the metric it was
    actually about went 4 to 30. Noise has to be as unable to manufacture a
    failure as it is to manufacture a success, so the floor is the first of four
    conditions here rather than the only one.

    The note comes back either way. A guardrail that crossed its floor without
    clearing the evidence bar is still something the founder should see; what it
    is not is a verdict.
    """
    guardrail = row["guardrail_metric"] or "none"
    floor = row["guardrail_floor"]
    baseline = row["guardrail_baseline"]
    if guardrail == "none" or floor is None or baseline is None:
        return (False, None)
    current = snapshot.resolve_metric(snap, guardrail)
    if current is None or not _breached(current, floor, POLARITY.get(guardrail, 1)):
        return (False, None)

    crossed = "guardrail %s is past its floor (%.4g, was %.4g)" % (guardrail, current, baseline)
    if guardrail in NON_COUNT_METRICS:
        return (False, "%s, and is a stock, a share or a bounded index the count test "
                       "does not apply to — not called harmful on that alone" % crossed)
    age = _age_days(row["created_at"]) or 0.0
    if age < window_days(guardrail):
        return (False, "%s, but its two readings are %.1f days apart against a %d-day "
                       "window and are largely the same events"
                       % (crossed, age, window_days(guardrail)))
    if not stats.gradeable_count(int(round(baseline))):
        return (False, "%s, but a baseline of %d is under the %d events a verdict needs"
                       % (crossed, int(round(baseline)), stats.MIN_VERDICT_EVENTS))
    pvalue = _pvalue(baseline, current)
    if pvalue > ALPHA:
        return (False, "%s, but p=%.3f — not separable from noise" % (crossed, pvalue))
    return (True, "%s, p=%.3f" % (crossed, pvalue))


def _write_verdict(
    row_id: int,
    verdict: str,
    pvalue: Optional[float],
    current: Optional[float],
    enacted: int,
    evidence: Optional[str],
) -> None:
    """Commit. `outcome` and `reviewed_at` are kept in step because snapshot.py
    reads `outcome` — which is how the verdict reaches the prompt as a given."""
    db.connect().execute(
        "UPDATE recommendations SET verdict=?, verdict_at=?, verdict_pvalue=?, "
        "verdict_current=?, verdict_source=?, enacted=?, enactment_evidence=?, "
        "outcome=?, reviewed_at=? WHERE id=?",
        (
            verdict,
            db.now(),
            pvalue,
            current,
            "scorecard",
            enacted,
            evidence,
            verdict,
            db.now(),
            row_id,
        ),
    )


def _revoke(row_id: int, current: Optional[float]) -> None:
    """Withdraw a `worked` that did not hold. Everything moves together.

    Writing only verdict_revoked_at left the row saying `worked`, beside a
    p-value from a comparison that no longer describes anything on it — the
    founder read "verdict: worked (moved as predicted) ... p=0.012 ... revoked
    on re-check" against a delta of zero. And `outcome` is the column snapshot.py
    feeds back to the model as past_recommendations, so an untouched outcome
    means the model is told this one worked for as long as the row exists.
    """
    stamp = db.now()
    db.connect().execute(
        "UPDATE recommendations SET verdict=?, verdict_at=?, verdict_pvalue=NULL, "
        "verdict_current=?, verdict_revoked_at=?, outcome=?, reviewed_at=? WHERE id=?",
        ("revoked", stamp, current, stamp, "revoked", stamp, row_id),
    )


def _recheck(guild_id: int, snap: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Revisit closed `worked` rows once more at twice the horizon.

    A number that crossed a line and fell straight back over it did not stay
    crossed, and a scorecard that never looks again would keep counting it.
    Only rows whose state changes are returned, so a stable win is not
    re-reported every week.
    """
    revoked: List[Dict[str, Any]] = []
    for row in db.query(
        "SELECT * FROM recommendations WHERE guild_id=? AND verdict='worked' "
        "AND verdict_revoked_at IS NULL",
        (guild_id,),
    ):
        age = _age_days(row["created_at"]) or 0.0
        if age < 2 * _horizon(row):
            continue
        current = snapshot.resolve_metric(snap, row["metric"] or "none")
        if current is None:
            continue
        threshold = row["threshold"]
        baseline = row["baseline"]
        if threshold is None or baseline is None:
            continue
        if _cleared(current, baseline, threshold, row["direction"] or "unchanged"):
            continue
        _revoke(row["id"], current)
        report = _as_report(db.one("SELECT * FROM recommendations WHERE id=?", (row["id"],)))
        report["note"] = (
            "revoked at %d days: the metric fell back below its threshold" % (2 * _horizon(row))
        )
        report["newly_closed"] = True
        revoked.append(report)
    return revoked


def score(guild_id: int, snap: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Grade everything past its horizon, and re-check old wins. Commits.

    Returns one report dict per row that changed state in this pass. No model is
    consulted, so this cannot hang, and the caller can commit before it risks a
    generation that might.
    """
    ensure_schema()
    if snap is None:
        snap = snapshot.build(guild_id)
    closed = [_grade(row, snap) for row in _due(guild_id)]
    return closed + _recheck(guild_id, snap)


def record_narration(refs: List[str], backend_label: str) -> None:
    """Note which backend wrote the sentence explaining a verdict. Once, ever.

    Separate from the verdict itself, which no backend produced.

    The WHERE clause is what makes it once: every path that renders a brief
    reads closed rows back for context, and without it a row's verdict_source
    was rewritten to whichever model most recently happened to look at it —
    months after the sentence it names was written.
    """
    ensure_schema()
    conn = db.connect()
    for item in refs:
        try:
            row_id = int(item.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        conn.execute(
            "UPDATE recommendations SET verdict_source=? "
            "WHERE id=? AND verdict_source='scorecard'",
            ("scorecard; narrated by %s" % backend_label, row_id),
        )


def rows_for_cli(guild_id: int, limit: int = 30) -> List[Dict[str, Any]]:
    """Every recent row, open or closed, with the numbers the CLI prints."""
    ensure_schema()
    return [
        _as_report(row)
        for row in db.query(
            "SELECT * FROM recommendations WHERE guild_id=? ORDER BY created_at DESC LIMIT ?",
            (guild_id, limit),
        )
    ]
