"""Grading of pre-registered recommendations. Deterministic, no model.

snapshot.py is what stops cadybot hallucinating a member count. This module is
what stops it hallucinating that its advice helped. Same rule, applied to
judgments instead of statistics: every value below is computed from the database
and from the snapshot, and nothing here may import advisor or llm or ask a model
anything. A verdict arrives at the prompt as a fact the model did not produce
and cannot revise.

A verdict of `worked` means: the founder demonstrably did something, and the
metric named *before* the fact crossed a threshold chosen *before* the fact, by
more than the noise in a count of that size. It does not mean the advice caused
the movement. One server, no control, no randomisation — attribution is not
available, and the word stays narrow on purpose.

Do not add a self-critique or "reconsider the verdict" pass, here or upstream.
Huang et al. (ICLR 2024) measured GPT-3.5 flipping 7.6% of its wrong GSM8K
answers right and 8.8% of its right ones wrong when asked to reconsider without
new information — a net loss. Every reflective step must be anchored to a number
the model did not produce, which is what this file exists to supply.
"""

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config, db, snapshot, stats

# The whole vocabulary. Every closed row holds exactly one of these.
VERDICTS: Tuple[str, ...] = (
    "worked",         # enacted, and the pre-registered metric cleared its threshold
    "failed",         # enacted, and it did not
    "harmful",        # a guardrail metric went through its floor
    "inconclusive",   # the move is not distinguishable from noise
    "not_attempted",  # no positive evidence the founder did the thing
    "unmeasurable",   # the recommendation named no snapshot metric
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

# Metrics that are counts of events, and so can be tested with the conditional
# binomial. A share and a day-count are neither Poisson nor comparable that way;
# rows naming one resolve inconclusive rather than borrowing a test that does
# not apply to them.
NON_COUNT_METRICS = frozenset(
    ["top_share_30d.top1.share", "top_share_30d.owner.share",
     "activity.days_since_owner_posted"]
)

# Snapshot paths recorded at issue time purely so enactment can be established
# later. Each one moves only if a human did something in the server.
ENACTMENT_PATHS: Tuple[str, ...] = (
    "activity.days_since_owner_posted",
    "activity.owner_messages_7d",
    "response_rate.answered",
    "channels.count",
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
    """Two-sided p for a change between two counts over equal windows.

    p0 is 0.5 because baseline and current are the same metric over the same
    window length; the exposures are equal by construction.
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
    if direction == "unchanged":
        return "%s holds within %d%% of %.4g for %d days" % (
            metric, int(MIN_EFFECT * 100), threshold or 0.0, horizon
        )
    return "%s goes %s, past %.4g, within %d days" % (metric, direction, threshold or 0.0, horizon)


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
    ensure_schema()
    return [
        row
        for row in db.query(
            "SELECT * FROM recommendations WHERE guild_id=? AND verdict IS NULL "
            "ORDER BY created_at",
            (guild_id,),
        )
        if (_age_days(row["created_at"]) or 0) >= _horizon(row)
    ]


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

    Every clause below is something that only becomes true when a human acts in
    the server. Nothing here reads "the founder stopped complaining" or "no harm
    appeared" as enactment: absence of evidence resolves to `not_attempted`,
    which is neutral, not success.

    The signals are generic rather than per-recommendation — cadybot cannot read
    a free-text action and go check it — so this gate is coarse. It is
    deliberately biased toward not_attempted, which is the safe direction: it
    costs a true positive rather than manufacturing one.
    """
    try:
        before = json.loads(row["baseline_json"] or "{}")
    except (TypeError, ValueError):
        before = {}
    age = _age_days(row["created_at"]) or 0.0
    now = _snapshot_values(snap, ENACTMENT_PATHS)

    silent = now.get("activity.days_since_owner_posted")
    if silent is not None and silent < age:
        return "the owner posted in the server %.0f days ago, after %s was issued" % (
            silent, ref(row["id"])
        )

    answered_before = before.get("response_rate.answered")
    answered_now = now.get("response_rate.answered")
    if answered_before is not None and answered_now is not None and answered_now > answered_before:
        return "the owner answered %d more member question(s) than at issue time" % (
            int(answered_now - answered_before)
        )

    channels_before = before.get("channels.count")
    channels_now = now.get("channels.count")
    if channels_before is not None and channels_now is not None and channels_now != channels_before:
        return "channel structure changed: %d channels, was %d" % (
            int(channels_now), int(channels_before)
        )

    threads_before = before.get("threads.opened_30d")
    threads_now = now.get("threads.opened_30d")
    if threads_before is not None and threads_now is not None and threads_now > threads_before:
        return "%d more thread(s) opened than at issue time" % int(threads_now - threads_before)

    return None


def _grade(row: Any, snap: Dict[str, Any]) -> Dict[str, Any]:
    """One row, one verdict. Precedence is fixed and is the whole design."""
    metric = row["metric"] or "none"
    baseline = row["baseline"]
    threshold = row["threshold"]
    direction = row["direction"] or "unchanged"
    current = snapshot.resolve_metric(snap, metric) if metric != "none" else None

    evidence = _enactment(snap, row)
    enacted = 1 if evidence else 0
    pvalue: Optional[float] = None
    note = ""

    if metric == "none" or metric not in snapshot.SCOREABLE_METRICS:
        verdict = "unmeasurable"
        note = "no snapshot metric was pre-registered"
    elif not enacted:
        # The enactment gate. `worked` needs the founder to have done the thing
        # AND the number to have moved; no evidence of the first makes the
        # second uninterpretable, however it went.
        verdict = "not_attempted"
        note = "no evidence in the snapshot that the action was taken"
    elif _guardrail_breached(row, snap):
        # Outranks the primary metric clearing its threshold on purpose: an
        # intervention that moved its target and broke something else is not a
        # success with a caveat.
        verdict = "harmful"
        note = "guardrail %s went through its floor" % row["guardrail_metric"]
    elif current is None or baseline is None:
        verdict = "inconclusive"
        note = "the metric is not present in the current snapshot"
    elif metric in NON_COUNT_METRICS:
        verdict = "inconclusive"
        note = "this metric is a share or an age, which the count test does not apply to"
    else:
        pvalue = _pvalue(baseline, current)
        if pvalue > ALPHA:
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

    _write_verdict(row["id"], verdict, pvalue, current, enacted, evidence)

    report = _as_report(
        db.one("SELECT * FROM recommendations WHERE id=?", (row["id"],))
    )
    report["note"] = note
    report["newly_closed"] = True
    return report


def _guardrail_breached(row: Any, snap: Dict[str, Any]) -> bool:
    guardrail = row["guardrail_metric"] or "none"
    floor = row["guardrail_floor"]
    if guardrail == "none" or floor is None:
        return False
    current = snapshot.resolve_metric(snap, guardrail)
    if current is None:
        return False
    return _breached(current, floor, POLARITY.get(guardrail, 1))


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
        db.connect().execute(
            "UPDATE recommendations SET verdict_revoked_at=?, verdict_current=? WHERE id=?",
            (db.now(), current, row["id"]),
        )
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
    """Note which backend wrote the sentence explaining a verdict.

    Separate from the verdict itself, which no backend produced.
    """
    ensure_schema()
    conn = db.connect()
    for item in refs:
        try:
            row_id = int(item.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        conn.execute(
            "UPDATE recommendations SET verdict_source=? WHERE id=?",
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
