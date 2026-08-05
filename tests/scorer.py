"""Grading harness. Deterministic, no model, no Discord.

tests/harness.py checks that cadybot reads a server correctly. This one checks
that it grades *itself* correctly, which is the other half and the easier one to
get wrong: every failure here is a scorer that reports good news it has not
earned, and a scorer that cannot report bad news is worse than no scorer.

Snapshots are hand-built dicts rather than synthesised servers. What is under
test is the arithmetic between a stored baseline and a current reading, so the
numbers are written down directly and the case is the exact one being argued
about.

Run:
    .venv/bin/python tests/scorer.py
"""

import os
import pathlib
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "scorer.db"
os.environ["CADYBOT_DB"] = "scorer.db"
os.environ.pop("GUILD_ID", None)
os.environ.pop("OWNER_ID", None)

from pydantic import ValidationError  # noqa: E402

from cadybot import advisor, db, scorecard  # noqa: E402

# The owner acted, or did not, in the way _enactment is allowed to read. Every
# case that is about something else carries one of these so the enactment gate
# is not what is being measured.
ACTED = {"activity.owner_messages_7d": 9, "channels.count": 3}
IDLE = {"activity.owner_messages_7d": 2, "channels.count": 3}

CASES = []


def case(name, metric, direction, horizon, before, after, expect, guardrail="none", at=None):
    CASES.append(
        {"name": name, "metric": metric, "direction": direction, "horizon": horizon,
         "before": before, "after": after, "expect": expect, "guardrail": guardrail,
         "at": at}
    )


def snap(base, **values):
    out = {}
    merged = dict(base)
    merged.update(values)
    for path, value in merged.items():
        node = out
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


# --- C1: enactment is a change in what the owner does ----------------------

# A founder who posts every day satisfies "the owner has posted recently" on the
# day the row is written and on every day after it. That is not evidence.
case("busy owner, no change", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300, "activity.owner_messages_7d": 7,
                   "activity.days_since_owner_posted": 0.5}),
     snap(IDLE, **{"activity.messages_30d": 450, "activity.owner_messages_7d": 7,
                   "activity.days_since_owner_posted": 0.5}),
     "not_attempted")

# Threads and answered questions are things any member does while the owner is
# silent, so neither may open the gate.
case("member opened a thread", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300, "activity.owner_messages_7d": 0,
                   "threads.opened_30d": 0, "activity.days_since_owner_posted": 45}),
     snap(IDLE, **{"activity.messages_30d": 450, "activity.owner_messages_7d": 0,
                   "threads.opened_30d": 1, "activity.days_since_owner_posted": 52}),
     "not_attempted")
case("member answered a question", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300, "activity.owner_messages_7d": 0,
                   "response_rate.answered": 2, "activity.days_since_owner_posted": 45}),
     snap(IDLE, **{"activity.messages_30d": 450, "activity.owner_messages_7d": 0,
                   "response_rate.answered": 5, "activity.days_since_owner_posted": 52}),
     "not_attempted")
case("owner posted more than before", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300}),
     snap(ACTED, **{"activity.messages_30d": 450}),
     "worked")

# --- C2: the direction has to be the one that improves the server ----------

case("more members go quiet", "members.gone_quiet.count", "up", 14,
     snap(IDLE, **{"members.gone_quiet.count": 2}),
     snap(ACTED, **{"members.gone_quiet.count": 20}),
     "unmeasurable")
case("more members leave", "membership_flow_30d.leaves", "up", 30,
     snap(IDLE, **{"membership_flow_30d.leaves": 30}),
     snap(ACTED, **{"membership_flow_30d.leaves": 90}),
     "unmeasurable")

# --- C3: no verdict under a baseline that moves on its own -----------------

for baseline, current in ((1, 8), (2, 10), (3, 12), (0, 6), (4, 15), (2, 9)):
    case("seven-member server %d->%d" % (baseline, current), "activity.messages_7d", "up", 7,
         snap(IDLE, **{"activity.messages_7d": baseline}),
         snap(ACTED, **{"activity.messages_7d": current}),
         "inconclusive")

# --- C4: two readings have to be a full window apart -----------------------

case("30-day metric judged in a week", "activity.messages_30d", "up", 7,
     snap(IDLE, **{"activity.messages_30d": 300}),
     snap(ACTED, **{"activity.messages_30d": 390}),
     "inconclusive", at=7.5)
case("a stock is not a Poisson count", "members.humans", "up", 30,
     snap(IDLE, **{"members.humans": 40}),
     snap(ACTED, **{"members.humans": 62}),
     "inconclusive")
case("a bounded index is not one either", "bus_factor_30d.factor", "up", 30,
     snap(IDLE, **{"bus_factor_30d.factor": 20}),
     snap(ACTED, **{"bus_factor_30d.factor": 34}),
     "inconclusive")

# --- C5: a guardrail needs the evidence a success needs --------------------

case("guardrail noise", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300, "members.gone_quiet.count": 2}),
     snap(ACTED, **{"activity.messages_30d": 450, "members.gone_quiet.count": 4}),
     "worked", guardrail="members.gone_quiet.count")
case("guardrail really broke", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300, "membership_flow_30d.leaves": 20}),
     snap(ACTED, **{"activity.messages_30d": 450, "membership_flow_30d.leaves": 60}),
     "harmful", guardrail="membership_flow_30d.leaves")

# --- C6: unchanged is an equivalence claim, and is gradeable ---------------

case("held, at a count that could tell", "activity.messages_30d", "unchanged", 30,
     snap(IDLE, **{"activity.messages_30d": 300}),
     snap(ACTED, **{"activity.messages_30d": 310}),
     "worked")
case("did not hold", "activity.messages_30d", "unchanged", 30,
     snap(IDLE, **{"activity.messages_30d": 300}),
     snap(ACTED, **{"activity.messages_30d": 700}),
     "failed")
case("held, but nothing could be told", "activity.messages_30d", "unchanged", 30,
     snap(IDLE, **{"activity.messages_30d": 25}),
     snap(ACTED, **{"activity.messages_30d": 26}),
     "inconclusive")

# --- the ordinary failure, which must still be reachable -------------------

case("did not move", "activity.messages_30d", "up", 30,
     snap(IDLE, **{"activity.messages_30d": 300}),
     snap(ACTED, **{"activity.messages_30d": 150}),
     "failed")


def issue(gid, snapshot_at_issue, metric, direction, horizon, guardrail, age_days):
    ids = scorecard.pre_register(
        gid, snapshot_at_issue,
        [{"headline": "h", "action": "a", "evidence": "e", "metric": metric,
          "direction": direction, "horizon_days": horizon, "guardrail_metric": guardrail}],
    )
    db.connect().execute(
        "UPDATE recommendations SET created_at=?, horizon_days=? WHERE id=?",
        ((datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat(), horizon, ids[0]),
    )
    return ids[0]


def run():
    for suffix in ("", "-wal", "-shm"):
        stale = pathlib.Path(str(DB) + suffix)
        if stale.exists():
            stale.unlink()

    passed = failed = 0
    gid = 900000

    for c in CASES:
        gid += 1
        at = c["at"] if c["at"] is not None else c["horizon"] + 0.5
        issue(gid, c["before"], c["metric"], c["direction"], c["horizon"], c["guardrail"], at)
        reports = scorecard.score(gid, c["after"])
        got = reports[0]["verdict"] if reports else "nothing graded"
        if got == c["expect"]:
            passed += 1
            print("ok    %-34s %s" % (c["name"], got))
        else:
            failed += 1
            print("FAIL  %-34s got %s, wanted %s" % (c["name"], got, c["expect"]))
            if reports:
                print("        %s" % reports[0].get("note"))

    # C7: a revoked verdict has to read as revoked everywhere, including in the
    # `outcome` column snapshot.py feeds back to the model.
    gid += 1
    before = snap(IDLE, **{"activity.messages_30d": 300})
    row_id = issue(gid, before, "activity.messages_30d", "up", 30, "none", 31)
    scorecard.score(gid, snap(ACTED, **{"activity.messages_30d": 450}))
    db.connect().execute(
        "UPDATE recommendations SET created_at=? WHERE id=?",
        ((datetime.now(timezone.utc) - timedelta(days=61)).isoformat(), row_id),
    )
    scorecard.score(gid, snap(ACTED, **{"activity.messages_30d": 300}))
    row = db.one("SELECT verdict, outcome, verdict_pvalue FROM recommendations WHERE id=?",
                 (row_id,))
    checks = [
        ("revocation clears the verdict", row["verdict"] == "revoked"),
        ("revocation clears the outcome", row["outcome"] == "revoked"),
        ("revocation drops the stale p", row["verdict_pvalue"] is None),
    ]

    # C4 and the mediums, which are properties of one call rather than verdicts.
    gid += 1
    ids = scorecard.pre_register(
        gid, before,
        [{"headline": "h", "action": "a", "evidence": "e", "metric": "activity.messages_30d",
          "direction": "up", "horizon_days": 7, "guardrail_metric": "none"}],
    )
    stored = db.one("SELECT horizon_days, prediction FROM recommendations WHERE id=?", (ids[0],))
    checks.append(("horizon is raised to the metric's window", stored["horizon_days"] == 30))
    checks.append(("the promise quotes the raised horizon", "30 days" in stored["prediction"]))

    gid += 1
    ids = scorecard.pre_register(
        gid, before,
        [{"headline": "h", "action": "a", "evidence": "e",
          "metric": "voice_30d.unique_participants", "direction": "up",
          "horizon_days": 14, "guardrail_metric": "none"}],
    )
    absent = db.one("SELECT prediction FROM recommendations WHERE id=?", (ids[0],))
    checks.append(("no bar is promised for an absent metric",
                   "past 0" not in absent["prediction"]))

    db.connect().execute(
        "UPDATE recommendations SET created_at='not a date' WHERE id=?", (ids[0],))
    broken = scorecard.score(gid, snap(ACTED))
    checks.append(("an unparseable created_at surfaces",
                   bool(broken) and broken[0]["verdict"] == "unmeasurable"))

    db.connect().execute(
        "UPDATE recommendations SET verdict='worked', verdict_source='scorecard' WHERE id=?",
        (ids[0],))
    scorecard.record_narration([scorecard.ref(ids[0])], "first")
    scorecard.record_narration([scorecard.ref(ids[0])], "second")
    source = db.one("SELECT verdict_source FROM recommendations WHERE id=?", (ids[0],))[0]
    checks.append(("a row is narrated once", source.endswith("first")))

    # The schema is the first line of defence for the polarity rule; the grader
    # above is the second, for rows written before it existed.
    rec = {"evidence": "e", "reasoning": "r", "would_change_my_mind": "w",
           "play_fails_when": "f", "headline": "h", "action": "a",
           "metric": "members.gone_quiet.count", "direction": "up",
           "horizon_days": 14, "guardrail_metric": "none"}
    try:
        advisor.Recommendation(**rec)
        checks.append(("a self-defeating bet is refused at issue time", False))
    except ValidationError:
        checks.append(("a self-defeating bet is refused at issue time", True))

    checks.append((
        "the brief names its conclusion last",
        list(advisor.Brief.model_json_schema()["properties"])[-1] == "headline",
    ))

    for name, ok in checks:
        if ok:
            passed += 1
            print("ok    %s" % name)
        else:
            failed += 1
            print("FAIL  %s" % name)

    print("\n%d passed, %d failed" % (passed, failed))
    for suffix in ("", "-wal", "-shm"):
        stale = pathlib.Path(str(DB) + suffix)
        if stale.exists():
            stale.unlink()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
