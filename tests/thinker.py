"""The desk. Deterministic, no model, no Discord.

tests/harness.py checks cadybot reads a server correctly; tests/scorer.py checks
it grades itself correctly. This one checks it knows when to *think*, which is
the half where the failure is not a wrong answer but a bot that talks forever
about nothing.

Most of these cases are regressions against bugs that were caught in design
review rather than in production, and they are written to stay caught. The three
that matter most:

  - `provoked_by` is always in the past (case 10). Three separate designs for
    this feature reached for the clock, which makes the "has something happened"
    check a tautology and the bot a perpetual motion machine.
  - `activity.days_since_owner_posted` can never be watched (case 6). It rises
    by 1.0 a day *because* nobody posts, so anything watching it fires forever
    on a dead server.
  - Two metrics drifting off the same event produce one thought, not two
    (case 14).

Run:
    .venv/bin/python tests/thinker.py
"""

import json
import os
import pathlib
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "thinker.db"
os.environ["CADYBOT_DB"] = "thinker.db"
os.environ.pop("GUILD_ID", None)
os.environ.pop("OWNER_ID", None)

from cadybot import agenda, config, db, ledger, scorecard, snapshot  # noqa: E402

# The verdict columns live on scorecard's own migration, not in db.SCHEMA, so a
# fresh test database has a `recommendations` table without them until this runs.
db.connect()
scorecard.ensure_schema()

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print("ok    %s" % name)
    else:
        FAILED.append(name)
        print("FAIL  %s   %s" % (name, detail))


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat()


def ago(**kw):
    return iso(datetime.now(timezone.utc) - timedelta(**kw))


GID = 900100


def fresh(gid=GID, consume_context=True):
    """A guild with nothing in it and an agenda that opened a year ago.

    `consume_context` journals the context-file provocation before handing the
    guild back. The repo's own context/*.md files have a real mtime, so on any
    fixture whose agenda opened before that date the context generator fires —
    correctly, and ahead of `drift` in precedence, masking whatever the test was
    actually about. Consuming it first is exactly what the first live tick does.
    """
    conn = db.connect()
    for table in db.GUILD_TABLES:
        conn.execute("DELETE FROM %s WHERE guild_id=?" % table, (gid,))
    conn.execute(
        "INSERT INTO guilds (guild_id, name, first_seen) VALUES (?, ?, ?)",
        (gid, "test", ago(days=400)),
    )
    db.set_setting(gid, agenda.INSTALLED_KEY, ago(days=365))
    if consume_context:
        prov = agenda._from_context(gid, {}, ago(days=365))
        if prov is not None:
            agenda.open_attempt(gid, prov)
    return gid


def member(gid, uid, is_bot=0, name="m"):
    db.connect().execute(
        "INSERT OR REPLACE INTO members (guild_id, user_id, username, display_name, "
        "is_bot, first_seen) VALUES (?, ?, ?, ?, ?, ?)",
        (gid, uid, name, name, is_bot, ago(days=300)),
    )


def message(gid, mid, uid, when, channel=5000):
    db.connect().execute(
        "INSERT OR REPLACE INTO messages (guild_id, channel_id, message_id, author_id, "
        "created_at, content, type) VALUES (?, ?, ?, ?, ?, ?, 0)",
        (gid, channel, mid, uid, when, "hello"),
    )


def close_day(gid, day, metric, value):
    db.connect().execute(
        "INSERT OR REPLACE INTO ledger (guild_id, day, metric, value) VALUES (?, ?, ?, ?)",
        (gid, day, metric, value),
    )


def empty_snap():
    return {"activity": {"messages_30d": 0}}


# ---------------------------------------------------------------- structural

src = {
    name: (ROOT / "cadybot" / ("%s.py" % name)).read_text()
    for name in ("agenda", "ledger", "thinking", "advisor")
}


def imports(text):
    out = []
    for line in text.split("\n"):
        if line.startswith("from . import"):
            out += [p.strip() for p in line.split("import", 1)[1].split(",")]
        elif line.startswith("from . import (") or line.strip().startswith(") "):
            pass
    # multi-line "from . import (\n a, b,\n)"
    if "from . import (" in text:
        block = text.split("from . import (", 1)[1].split(")", 1)[0]
        out += [p.strip() for p in block.replace("\n", " ").split(",")]
    return set(p for p in out if p)


check(
    "1  agenda imports no model",
    not (imports(src["agenda"]) & {"advisor", "llm"}),
    sorted(imports(src["agenda"])),
)
check(
    "1b ledger imports no model",
    not (imports(src["ledger"]) & {"advisor", "llm"}),
    sorted(imports(src["ledger"])),
)
def calls_score(text):
    """Does this module actually reach for `*.score(...)` anywhere in its code?

    Walking the AST rather than grepping, because thinking.py's own docstring
    explains at length why it must never call scorecard.score — a substring
    search finds the prohibition and reports it as the violation.
    """
    import ast

    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "score":
                return True
    return False


check(
    "2  thinking never grades",
    "scorecard" not in imports(src["thinking"]) and not calls_score(src["thinking"]),
    "thinking.py must not call scorecard.score — a second grader eats the "
    "verdict loop.nightly was going to narrate",
)
check(
    "3  the import arrow points one way",
    "agenda" in imports(src["advisor"]) and "advisor" not in imports(src["agenda"]),
)
check(
    "4  the desk never touches loop's baseline",
    all(
        "loop_last_report" not in src[m] and "LAST_REPORT_KEY" not in src[m]
        for m in ("agenda", "ledger", "thinking")
    ),
)

from cadybot import advisor  # noqa: E402

props = list(advisor.Reflection.model_json_schema()["properties"])
check("5  reflection names its conclusion last", props[-1] == "to_founder", props)
check(
    "6  the clock metric cannot be watched",
    "activity.days_since_owner_posted" not in advisor.COUNT_METRIC_CHOICES
    and not (set(advisor.COUNT_METRIC_CHOICES) & set(scorecard.NON_COUNT_METRICS)),
    "a metric that moves because time passed is a clock, not news",
)

# --------------------------------------------------------------- provocation

gid = fresh()
check("7  nothing in it, nothing to think about", agenda.next_provocation(gid, empty_snap()) is None)
check("7b installing writes the mark", bool(db.get_setting(gid, agenda.INSTALLED_KEY)))

gid = fresh()
member(gid, 1, name="human")
message(gid, 10, 1, ago(days=400))
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 0.0)
check("8  history is not news", agenda.next_provocation(gid, empty_snap()) is None)

gid = fresh()
db.connect().execute(
    "INSERT INTO recommendations (guild_id, created_at, headline, action, metric, "
    "prediction, verdict, verdict_at, baseline, threshold, verdict_current) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    (gid, ago(days=20), "h", "do the thing", "activity.messages_7d",
     "goes up past 1 in 7 days", "not_attempted", ago(hours=3), 0.0, 1.0, 0.0),
)
rid = db.scalar("SELECT id FROM recommendations WHERE guild_id=?", (gid,))
prov = agenda.next_provocation(gid, empty_snap())
check("9  a closed verdict provokes", prov is not None and prov.kind == "verdict")
check("9b it points at the row", prov and prov.about_ref == scorecard.ref(rid))
check(
    "9c provoked_by is the stored verdict_at",
    prov and prov.provoked_by == db.one(
        "SELECT verdict_at FROM recommendations WHERE id=?", (rid,))["verdict_at"],
)
check(
    "9d the prompt carries the action verbatim",
    prov and "do the thing" in prov.self_prompt,
)

# ----------------------------------------------------- 10: the tautology test

seen_kinds = []


def build_every_kind():
    """One fixture per generator, so the past-tense assertion runs on all five."""
    out = []

    g = fresh(900201)
    db.connect().execute(
        "INSERT INTO recommendations (guild_id, created_at, headline, action, metric, "
        "verdict, verdict_at, baseline, threshold, verdict_current) "
        "VALUES (?, ?, 'h', 'a', 'activity.messages_7d', 'failed', ?, 0.0, 1.0, 0.0)",
        (g, ago(days=20), ago(hours=2)),
    )
    out.append(g)

    g = fresh(900202)
    member(g, 1, name="human")
    close_day(g, ledger.day_offset(1), "activity.messages_30d", 0.0)
    message(g, 11, 1, ago(hours=4))
    out.append(g)

    g = fresh(900203)
    db.connect().execute(
        "INSERT INTO member_events (guild_id, user_id, event, at) VALUES (?, ?, 'join', ?)",
        (g, 7, ago(hours=6)),
    )
    out.append(g)

    g = fresh(900204)
    member(g, 1, name="human")
    message(g, 12, 1, ago(days=2))
    close_day(g, ledger.day_offset(ledger.DRIFT_DAYS), "activity.messages_30d", 0.0)
    close_day(g, ledger.today(), "activity.messages_30d", 40.0)
    out.append(g)
    return out


for g in build_every_kind():
    p = agenda.next_provocation(g, empty_snap())
    if p:
        seen_kinds.append(p.kind)
        check(
            "10 %-8s provoked_by is in the past" % p.kind,
            p.provoked_by < db.now(),
            p.provoked_by,
        )

check(
    "10b every generator except context was exercised",
    set(seen_kinds) >= {"verdict", "life", "joined", "drift"},
    sorted(set(seen_kinds)),
)

# 10c: clock skew must cost a cycle, not the whole desk. Discord stamps message
# timestamps server-side, so a host clock a few seconds slow makes the newest
# message look like it has not happened yet. This used to be an assertion, and
# it crashed the pass on every tick for as long as that message stayed newest.
gid = fresh(900205)
member(gid, 1, name="human")
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 0.0)
message(gid, 40, 1, iso(datetime.now(timezone.utc) + timedelta(seconds=30)))
try:
    skewed = agenda.next_provocation(gid, empty_snap())
    ok10c = skewed is None
    detail = skewed.kind if skewed else ""
except AssertionError as exc:
    ok10c, detail = False, "crashed: %s" % exc
check("10c a clock a few seconds out costs one tick, not the desk", ok10c, detail)

message(gid, 40, 1, ago(hours=2))   # the same row, once the clock agrees
check(
    "10d and the thought is not lost, only deferred",
    (agenda.next_provocation(gid, empty_snap()) or None) is not None,
)

# ------------------------------------------------------------------ life

gid = fresh()
member(gid, 1, name="human")
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 0.0)
message(gid, 20, 1, ago(hours=5))
prov = agenda.next_provocation(gid, empty_snap())
check("11 silence then a message provokes", prov is not None and prov.kind == "life")
check(
    "11b provoked_by is the message's own timestamp",
    prov and prov.provoked_by == db.one(
        "SELECT created_at FROM messages WHERE message_id=20", ())["created_at"],
)

gid = fresh()
member(gid, 1, name="human")
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 40.0)
message(gid, 21, 1, ago(hours=5))
check("11c a busy server is not woken", agenda.next_provocation(gid, empty_snap()) is None)

# 12: the ordering test — today's close must not erase yesterday's evidence
gid = fresh()
member(gid, 1, name="human")
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 0.0)
message(gid, 22, 1, ago(hours=5))
close_day(gid, ledger.today(), "activity.messages_30d", 1.0)   # today's pass ran first
prov = agenda.next_provocation(gid, empty_snap())
check("12 today's close cannot erase yesterday's silence", prov is not None and prov.kind == "life")

# ------------------------------------------------------------------ joined

gid = fresh()
db.connect().execute(
    "INSERT INTO member_events (guild_id, user_id, event, at) VALUES (?, 7, 'join', ?)",
    (gid, ago(days=3)),
)
db.connect().execute(
    "INSERT INTO member_events (guild_id, user_id, event, at) VALUES (?, 8, 'join', ?)",
    (gid, ago(hours=2)),
)
check("13 routine joins are weather", agenda.next_provocation(gid, empty_snap()) is None)

gid = fresh()
db.connect().execute(
    "INSERT INTO member_events (guild_id, user_id, event, at) VALUES (?, 8, 'join', ?)",
    (gid, ago(hours=2)),
)
prov = agenda.next_provocation(gid, empty_snap())
check("13b a join after a drought provokes", prov is not None and prov.kind == "joined")

# ------------------------------------------------------------------- drift

gid = fresh()
member(gid, 1, name="human")
message(gid, 30, 1, ago(days=2))
for metric in ("activity.messages_30d", "structure.mentions_30d"):
    close_day(gid, ledger.day_offset(ledger.DRIFT_DAYS), metric, 0.0)
    close_day(gid, ledger.today(), metric, 40.0)
prov = agenda.next_provocation(gid, empty_snap())
check("14 two metrics off one event is one thought", prov is not None and prov.kind == "drift")
jid = agenda.open_attempt(gid, prov)
check("14b and the second look finds nothing new", agenda.next_provocation(gid, empty_snap()) is None)

gid = fresh()
for metric in ("activity.messages_30d",):
    close_day(gid, ledger.day_offset(ledger.DRIFT_DAYS), metric, 40.0)
    close_day(gid, ledger.today(), metric, 0.0)
check("15 decay alone never provokes", agenda.next_provocation(gid, empty_snap()) is None)

# ----------------------------------------------------------------- context

gid = fresh(consume_context=False)
prov = agenda.next_provocation(gid, empty_snap())
check(
    "16 an edited context file provokes",
    prov is not None and prov.kind == "context",
    prov.kind if prov else None,
)
if prov:
    allowed, why = agenda.may_surface(gid, prov)
    check("16b and can never be surfaced", allowed is False and "never surfaced" in why, why)

# --------------------------------------------------------- budget and dedupe

gid = fresh()
p = agenda.Provocation("verdict", ago(hours=2), "prompt")
first = agenda.open_attempt(gid, p)
check("17 a provocation is claimed once", first is not None)
check("17b and cannot be claimed twice", agenda.open_attempt(gid, p) is None)

gid = fresh()
twin_a = agenda.Provocation("drift", ago(hours=2), "prompt")
twin_b = agenda.Provocation("drift", twin_a.provoked_by, "prompt built by another process")
check("18 racing processes collide by construction", agenda.open_attempt(gid, twin_a) is not None)
check("18b the loser loses", agenda.open_attempt(gid, twin_b) is None)

gid = fresh()
for n in range(config.THINK_CALLS_PER_DAY):
    jid = agenda.open_attempt(gid, agenda.Provocation("drift", ago(hours=n + 1), "p"))
    agenda.record_failure(jid, RuntimeError("model timed out"))
check("19 failures are charged", agenda.affordable(gid) is False)

db.connect().execute(
    "UPDATE journal SET started_at=? WHERE guild_id=?", (ago(days=2), gid)
)
check("20 yesterday's spend does not bind today", agenda.affordable(gid) is True)

_saved = config.THINK_CALLS_PER_DAY
config.THINK_CALLS_PER_DAY = 0
check("21 zero disables thinking entirely", agenda.affordable(fresh()) is False)
config.THINK_CALLS_PER_DAY = _saved

# ---------------------------------------------------------- the surface gate

gid = fresh()
surf = agenda.Provocation("verdict", ago(hours=2), "p")
_backend = config.BACKEND

# The allowlist is pinned explicitly rather than read from the environment: the
# deployment that actually runs this sets CADYBOT_THINK_SURFACE_BACKENDS to
# include ollama, and a test that inherits it asserts the local policy instead
# of the mechanism.
_allow_before = config.THINK_SURFACE_BACKENDS
config.THINK_SURFACE_BACKENDS = ("anthropic",)
config.BACKEND = "ollama"
allowed, why = agenda.may_surface(gid, surf)
check("23 a backend off the allowlist may think but not speak",
      allowed is False and "may think" in why, why)
config.BACKEND = "anthropic"
config.THINK_SURFACE_BACKENDS = _allow_before

_window = config.SURFACE_WINDOW_UTC
config.SURFACE_WINDOW_UTC = (0, 24)
allowed, why = agenda.may_surface(gid, surf)
check("24 an open window allows it", allowed is True, why)

config.SURFACE_WINDOW_UTC = (23, 24) if datetime.now(timezone.utc).hour != 23 else (0, 1)
allowed, why = agenda.may_surface(gid, surf)
check("24b outside the window it waits", allowed is False and "window" in why, why)
config.SURFACE_WINDOW_UTC = (0, 24)

db.record_delivery(gid, "nightly", 100)
allowed, why = agenda.may_surface(gid, surf)
check("25 it will not talk over another message", allowed is False and "within the last" in why, why)
db.connect().execute("UPDATE deliveries SET at=? WHERE guild_id=?", (ago(hours=30), gid))
allowed, why = agenda.may_surface(gid, surf)
check("25b a day later it may", allowed is True, why)

_perweek = config.SURFACE_MAX_PER_WEEK
config.SURFACE_MAX_PER_WEEK = 1
db.connect().execute(
    "INSERT INTO deliveries (guild_id, at, kind, chars) VALUES (?, ?, 'thought', 10)",
    (gid, ago(days=3)),
)
allowed, why = agenda.may_surface(gid, surf)
check("26 the weekly ceiling holds", allowed is False and "this week" in why, why)
db.connect().execute(
    "UPDATE deliveries SET at=? WHERE guild_id=? AND kind='thought'", (ago(days=8), gid)
)
allowed, why = agenda.may_surface(gid, surf)
check("26b next week it may again", allowed is True, why)
config.SURFACE_MAX_PER_WEEK = _perweek

db.set_setting(gid, agenda.QUIET_KEY, iso(datetime.now(timezone.utc) + timedelta(days=2)))
allowed, why = agenda.may_surface(gid, surf)
check("27 /quiet silences it", allowed is False and "quiet until" in why, why)
db.set_setting(gid, agenda.QUIET_KEY, None)
config.SURFACE_WINDOW_UTC = _window
config.BACKEND = _backend

# --------------------------------------------------------------- the schema

from pydantic import ValidationError  # noqa: E402

BASE = dict(
    restated="q", reasoning="r", evidence="e",
    watch_metric="activity.messages_7d", worth_telling_founder=False,
)
# Dropped, not refused: raising costs the whole generation over the least
# important field. What matters is that nothing numeric is ever stored.
ok28 = advisor.Reflection(note_to_self="joins fell to 3 this week", **BASE).note_to_self == ""
check("28 a note carrying a number is discarded", ok28,
      "a number written down today is wrong in a month")
check(
    "28b the same note without one is fine",
    advisor.Reflection(note_to_self="joins fell this week", **BASE) is not None,
)

# 28c: a digit filter alone lets the identical failure through spelled out.
for spelled in ("Activity fell to four a week and joins to two.",
                "Roughly a third of members went quiet.",
                "About twelve people are still active.",
                "Half the server never posted."):
    if advisor.Reflection(note_to_self=spelled, **BASE).note_to_self != "":
        ok28c, why = False, spelled
        break
    ok28c, why = True, ""
check("28c nor a quantity written out in words", ok28c, why)

# and the note has to stay writable, or the model just stops writing them
for plain in ("Advice needing the founder to act is unmeasurable while he is away.",
              "The server is not the bottleneck; distribution is.",
              "One member asked a question and nobody answered."):
    try:
        advisor.Reflection(note_to_self=plain, **BASE)
        ok28d, why = True, ""
    except ValidationError as exc:
        ok28d, why = False, "%s -> %s" % (plain, exc)
        break
check("28d but an ordinary qualitative note still passes", ok28d, why)

try:
    advisor.Reflection(
        note_to_self="n", restated="q", reasoning="r", evidence="e",
        watch_metric="activity.days_since_owner_posted", worth_telling_founder=False,
    )
    ok29 = False
except ValidationError:
    ok29 = True
check("29 the clock metric is refused at the schema", ok29)

kept = advisor._drop_self_assessment(
    "My earlier note was right about this. Joins are flat."
)
check("30 self-assessment is stripped", "earlier note" not in kept, kept)

# 31: code-supplied provenance is not flagged as invented
prov = agenda.Provocation("verdict", ago(hours=2), "p", "R-1", agenda._numerals(1.0, 0.0))
snap31 = {"members": {"humans": 1}}
flagged = set(advisor.verify_evidence(snap31, "the threshold was 1 and the reading is 0"))
check(
    "31 cadybot is not flagged for quoting cadybot",
    not (flagged - agenda.known_numbers(prov)),
    sorted(flagged - agenda.known_numbers(prov)),
)

# --------------------------------------------------- the live server, a month

gid = fresh(900300)
member(gid, 1, is_bot=0, name="a_a_k")
for uid in (2, 3, 4, 5):
    member(gid, uid, is_bot=1, name="bot%d" % uid)
for n in range(88):
    message(gid, 1000 + n, 1, ago(days=60 + n % 30))
# The agenda opened after the backlog, as it does on a real install: everything
# already in the database on day one is history, not news. Without this the
# fixture is a server whose entire message history arrived overnight.
db.set_setting(gid, agenda.INSTALLED_KEY, ago(days=30))
db.connect().execute(
    "INSERT INTO recommendations (guild_id, created_at, headline, action, metric, "
    "direction, horizon_days, baseline, threshold) "
    "VALUES (?, ?, 'h', 'a', 'activity.messages_7d', 'up', 7, 0.0, 1.0)",
    (gid, ago(days=5)),
)
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 0.0)
first = agenda.next_provocation(gid, empty_snap())
check("32 the unread history is worth exactly one thought",
      first is not None and first.kind == "backlog", first.kind if first else None)
agenda.open_attempt(gid, first)
quiet = all(agenda.next_provocation(gid, empty_snap()) is None for _ in range(120))
check("32b and then a month of ticks produces none", quiet)

db.connect().execute(
    "UPDATE recommendations SET verdict='not_attempted', verdict_at=?, verdict_current=0.0 "
    "WHERE guild_id=?",
    (ago(hours=2), gid),
)
prov = agenda.next_provocation(gid, empty_snap())
check("33 closing the bet provokes exactly once", prov is not None and prov.kind == "verdict")
agenda.open_attempt(gid, prov)
check("33b and then goes quiet again",
      all(agenda.next_provocation(gid, empty_snap()) is None for _ in range(120)))

message(gid, 9999, 1, ago(hours=1))
prov = agenda.next_provocation(gid, empty_snap())
check("34 one message wakes it, once", prov is not None and prov.kind == "life")
agenda.open_attempt(gid, prov)
check("34b and only once",
      all(agenda.next_provocation(gid, empty_snap()) is None for _ in range(120)))

# ------------------------------------------------------------------ ledger

gid = fresh(900400)
member(gid, 1, name="human")
snap = snapshot.build(gid, owner_id=1)
ledger.record_day(gid, snap)
ledger.record_day(gid, snap)
rows = db.scalar("SELECT COUNT(*) FROM ledger WHERE guild_id=?", (gid,))
check("35 a day closes once, however often it runs", rows == len(ledger.LEDGER_METRICS), rows)
check(
    "36 every ledger metric resolves on a real snapshot",
    all(snapshot.resolve_metric(snap, m) is not None for m in ledger.LEDGER_METRICS),
    [m for m in ledger.LEDGER_METRICS if snapshot.resolve_metric(snap, m) is None],
)
close_day(gid, ledger.day_offset(ledger.LEDGER_DAYS + 5), "activity.messages_7d", 1.0)
before = db.scalar("SELECT COUNT(*) FROM ledger WHERE guild_id=?", (gid,))
ledger.prune(gid)
after = db.scalar("SELECT COUNT(*) FROM ledger WHERE guild_id=?", (gid,))
check("37 old closes are pruned, recent ones kept", before - after == 1, (before, after))

# -------------------------------------------------------------- deliveries

gid = fresh(900500)
db.record_delivery(gid, "nightly", 42)
check("38 a delivery is counted", db.deliveries_since(gid, ago(hours=1)) == 1)
check("38b and not before it happened", db.deliveries_since(gid, ago(seconds=-5)) == 0)
check("38c by kind", db.deliveries_since(gid, ago(hours=1), "weekly") == 0)

# ------------------------------------------------- the whole pass, end to end
#
# Everything above tests a part. This drives thinking.think itself with the
# model stubbed and a fake Discord client, which is the only way to check the
# things that only exist in the seams: that the journal row is written before
# the call and updated after, that a veto is honoured, that a delivery failure
# does not get recorded as a delivery, and that a crash still costs its budget.

import asyncio  # noqa: E402

from cadybot import notify, thinking  # noqa: E402

E2E, ROOM = 900600, 999
sent = []


class _Channel:
    id = ROOM
    name = "cadybot"

    def __init__(self, guild):
        self.guild = guild

    async def send(self, text):
        sent.append(text)


class _Guild:
    id = E2E

    def get_channel(self, cid):
        return _Channel(self) if cid == ROOM else None


class _Client:
    def get_guild(self, gid):
        return _Guild() if gid == E2E else None


class _Homeless:
    """A client whose guild is gone and whose owner cannot be reached."""

    def get_guild(self, gid):
        return None

    def get_user(self, uid):
        return None

    async def fetch_user(self, uid):
        return None


REPLY = dict(
    restated="Did the reading hold up?",
    reasoning="The bet was on messages rising and nothing moved.",
    evidence="messages_7d was 0 at issue and is 0 now.",
    note_to_self="Advice needing the founder to act is unmeasurable while he is away.",
    watch_metric="activity.messages_7d",
    worth_telling_founder=True,
    to_founder="Your last recommendation closed with no sign it was tried.",
)


def _last_journal():
    """The newest journal row for the e2e guild.

    Not `db.one(... WHERE guild_id=?)`: fresh() pre-consumes the context-file
    provocation, so the *first* row is always that one and every assertion below
    would be reading the wrong thought.
    """
    return db.one(
        "SELECT * FROM journal WHERE guild_id=? ORDER BY id DESC LIMIT 1", (E2E,)
    )


def _seed_desk():
    fresh(E2E)
    db.set_setting(E2E, "room_channel_id", str(ROOM))
    db.set_setting(E2E, "owner_id", "42")
    db.set_setting(E2E, agenda.INSTALLED_KEY, ago(days=2))
    db.connect().execute(
        "INSERT INTO recommendations (guild_id, created_at, headline, action, metric, "
        "prediction, verdict, verdict_at, baseline, threshold, verdict_current) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (E2E, ago(days=20), "h", "post the thing", "activity.messages_7d",
         "goes up past 1 in 7 days", "not_attempted", ago(hours=2), 0.0, 1.0, 0.0),
    )
    sent.clear()


_real_generate = advisor.llm.generate
_win, _backend = config.SURFACE_WINDOW_UTC, config.BACKEND
config.SURFACE_WINDOW_UTC, config.BACKEND = (0, 24), "anthropic"

_seed_desk()
advisor.llm.generate = lambda *a, **k: advisor.Reflection(**REPLY)
said = asyncio.run(thinking.think(_Client(), E2E))
row = _last_journal()
check("40 the whole pass runs and speaks", said is not None and len(sent) == 1)
check("40b it says why it spoke", "Prompted by a recommendation closed" in (said or ""))
check("40c the thought is journalled", row and row["outcome"] == "thought")
check("40d and marked as said", row and row["surfaced_at"] is not None)
check("40e and logged as a delivery", db.deliveries_since(E2E, ago(hours=1), "thought") == 1)
check("40f it will not say it twice", asyncio.run(thinking.think(_Client(), E2E)) is None)

_seed_desk()
advisor.llm.generate = lambda *a, **k: advisor.Reflection(
    **dict(REPLY, worth_telling_founder=False, to_founder=None)
)
said = asyncio.run(thinking.think(_Client(), E2E))
row = _last_journal()
check("41 the model's veto is obeyed", said is None and not sent)
check("41b the thinking is kept anyway", row and row["outcome"] == "thought")
check("41c and not counted as said", row and row["surfaced_at"] is None
      and db.deliveries_since(E2E, ago(hours=1)) == 0)

_seed_desk()
advisor.llm.generate = lambda *a, **k: advisor.Reflection(
    **dict(REPLY, to_founder="Engagement is down 73% this month.")
)
said = asyncio.run(thinking.think(_Client(), E2E))
row = _last_journal()
check("42 an invented number is never volunteered", said is None and not sent)
check("42b and is on the record", row and "73" in (row["unverified"] or ""))

_seed_desk()


def _boom(*a, **k):
    raise advisor.BackendError("the model went away")


advisor.llm.generate = _boom
try:
    asyncio.run(thinking.think(_Client(), E2E))
    raised = False
except advisor.BackendError:
    raised = True
row = _last_journal()
check("43 a failed call is not swallowed", raised)
check("43b it is recorded as failed, with the reason",
      row and row["outcome"] == "failed" and "went away" in (row["failure"] or ""))

_seed_desk()
db.set_setting(E2E, "owner_id", None)
advisor.llm.generate = lambda *a, **k: advisor.Reflection(**REPLY)
said = asyncio.run(thinking.think(_Homeless(), E2E))
row = _last_journal()
check("44 nowhere to say it means it was not said", said is None)
check("44b so it is not marked as said", row and row["surfaced_at"] is None
      and db.deliveries_since(E2E, ago(hours=1)) == 0)

advisor.llm.generate = _real_generate
config.SURFACE_WINDOW_UTC, config.BACKEND = _win, _backend

# ------------------------------------- regressions from the implementation review
#
# Every case below is a bug that shipped in the first cut of this feature and was
# found by adversarially reviewing the real diff. Four of them made the headline
# capability silently not work at all.

# 45: naming its own reference made every verdict thought unspeakable. R-1
# tokenises as '-1' under advisor._NUMERAL, which is in no snapshot, so the
# unverified gate suppressed it — the most valuable kind of thought, never said.
p45 = agenda.Provocation("verdict", ago(hours=2), "p", "R-1", agenda._numerals(0.0, 1.0))
left = set(advisor.verify_evidence({"members": {"humans": 1}},
                                   "R-1 closed without being tried.")) - agenda.known_numbers(p45)
check("45 it may name the bet it is thinking about", not left, sorted(left))

# 46: `life` used ORDER BY ASC against a fixed install mark, so it returned the
# first message ever posted after install — forever — and could fire once only.
gid = fresh(900701)
member(gid, 1, name="human")
close_day(gid, ledger.day_offset(1), "activity.messages_30d", 0.0)
message(gid, 60, 1, ago(days=200))
first = agenda.next_provocation(gid, empty_snap())
agenda.open_attempt(gid, first)
message(gid, 61, 1, ago(hours=2))          # months later, the room wakes again
second = agenda.next_provocation(gid, empty_snap())
check("46 a second reawakening still provokes", second is not None and second.kind == "life",
      second.kind if second else None)
check("46b and it is the new message, not the old one",
      second and second.provoked_by != first.provoked_by)

# 47: a join burst produced nothing, because each arrival cancelled the next.
gid = fresh(900702)
for uid, hours in ((1, 4), (2, 3), (3, 2)):
    db.connect().execute(
        "INSERT INTO member_events (guild_id, user_id, event, at) VALUES (?,?, 'join', ?)",
        (gid, uid, ago(hours=hours)))
burst = agenda.next_provocation(gid, empty_snap())
check("47 a burst of joins is news, not noise", burst is not None and burst.kind == "joined",
      burst.kind if burst else None)

# 48: drift's key was the newest message, so it re-provoked every tick forever
# while reporting the identical unchanged observation.
gid = fresh(900703)
member(gid, 1, name="human")
message(gid, 70, 1, ago(days=2))
close_day(gid, ledger.day_offset(ledger.DRIFT_DAYS), "activity.messages_30d", 0.0)
close_day(gid, ledger.today(), "activity.messages_30d", 40.0)
d1 = agenda.next_provocation(gid, empty_snap())
check("48 drift provokes once", d1 is not None and d1.kind == "drift")
agenda.open_attempt(gid, d1)
message(gid, 71, 1, ago(hours=1))          # somebody posts again
d2 = agenda.next_provocation(gid, empty_snap())
check("48b and not again on the next message", d2 is None or d2.kind != "drift",
      d2.kind if d2 else None)

# 49: a thought worth telling was destroyed when the tick fell outside the
# speaking window — three ticks in four.
gid = fresh(900704)
db.connect().execute(
    "INSERT INTO journal (guild_id, kind, provoked_by, about_ref, started_at, "
    "self_prompt, outcome, to_founder, note_to_self, watch_metric, unverified, "
    "wanted_telling) VALUES (?,?,?,?,?,?, 'thought', ?, ?, ?, '[]', 1)",
    (gid, "verdict", ago(hours=3), "R-9", ago(hours=3), "sp",
     "Your last bet closed untried.", "act on it", "activity.messages_7d"))
held = agenda.unsaid(gid)
check("49 a thought it could not say yet is kept", held is not None)
check("49b and can be rendered without a second model call",
      held is not None and "closed untried" in advisor.render_stored(held))
db.connect().execute("UPDATE journal SET surfaced_at=? WHERE guild_id=?", (db.now(), gid))
check("49c once said, it is not offered again", agenda.unsaid(gid) is None)

# 50: a SIGTERM during the model call left the row at 'started' forever, and the
# provocation could never be raised again.
gid = fresh(900705)
jid = agenda.open_attempt(gid, agenda.Provocation("verdict", ago(hours=5), "p"))
# open_attempt stamps started_at with now(); age it as a died-mid-call row would.
db.connect().execute("UPDATE journal SET started_at=? WHERE id=?", (ago(hours=5), jid))
check("50 an abandoned thought is reaped", agenda.reap_stale(gid) == 1)
check("50b and recorded as failed, not left open",
      db.one("SELECT outcome FROM journal WHERE id=?", (jid,))["outcome"] == "failed")
fresh_jid = agenda.open_attempt(fresh(900706),
                                agenda.Provocation("verdict", ago(minutes=5), "p"))
check("50c a thought still in flight is left alone", agenda.reap_stale(900706) == 0)

# 51: a note that was entirely self-assessment was restored uncleaned and
# replayed into briefs for sixty days.
check("51 an all-self-assessment note is dropped, not restored",
      advisor._drop_self_assessment("I was right about the DM push.") is None)

# 52: observed on the first real local run. The desk produced correct advice —
# go participate in r/3Dprinting instead of over-reading one message — and threw
# it away, because the numeral pattern read "3" out of "r/3Dprinting" and no
# snapshot contains a bare 3. The most likely sentence this advisor could write
# was the one that silenced it.
_snap52 = {"members": {"humans": 1},
           "activity": {"messages_7d": 0, "days_since_owner_posted": 87.8}}
for phrase in ("go participate in r/3Dprinting and Printables",
               "R-1 closed without being tried",
               "MakerWorld and Thingiverse are where they already are",
               "you have not posted in 87.8 days"):
    flagged = advisor.verify_evidence(_snap52, phrase)
    if flagged:
        break
check("52 a digit inside a word is not a cited number", not flagged, (phrase, flagged))
# 52c: a reflection that correctly said "nobody answered them on 2026-05-09"
# was flagged for inventing 2026, 05 and 09, and suppressed. Citing when
# something happened is the opposite of making a number up.
check("52c a date is not an invented statistic",
      not advisor.verify_evidence(_snap52, "nobody answered them on 2026-05-09 at 17:46"))
check("52b but a genuinely invented figure still is",
      advisor.verify_evidence(_snap52, "engagement is down 73% this month") == ["73"])

# 53: verify_evidence only ever looked at numbers, so an invented channel name
# passed every check cadybot had.
_snap53 = {"channels": {"count": 1, "all": [{"channel": "hermes"}]}}
check("53 an invented channel is caught",
      advisor.verify_entities(_snap53, "post in the #introductions channel") == ["introductions"])
check("53b and a real one is not",
      advisor.verify_entities(_snap53, "post in the 'hermes' channel") == [])

# 54: local inference may be allowed to speak, by configuration rather than by
# editing code.
_before = config.THINK_SURFACE_BACKENDS
config.THINK_SURFACE_BACKENDS = ("ollama", "anthropic")
config.BACKEND = "ollama"
_win = config.SURFACE_WINDOW_UTC
config.SURFACE_WINDOW_UTC = (0, 24)
_g = fresh(900800)
_ok, _why = agenda.may_surface(_g, agenda.Provocation("verdict", ago(hours=2), "p", "R-1"))
check("54 the local model may be allowed to speak", _ok, _why)
config.THINK_SURFACE_BACKENDS = ("anthropic",)
_ok2, _why2 = agenda.may_surface(_g, agenda.Provocation("verdict", ago(hours=2), "p", "R-1"))
check("54b and the default still refuses", not _ok2 and "may think" in _why2, _why2)
config.THINK_SURFACE_BACKENDS = _before
config.SURFACE_WINDOW_UTC = _win

# ------------------------------------------- plays: preconditions as arithmetic
#
# Observed on the live server: cadybot advised "immediately merge down all
# channels, leaving only one channel named `hermes`" on a server that has
# exactly one channel, already named hermes. The play's stated precondition is
# "more than about three channels exist" — prose, which the model walked
# straight through. These pin both halves: ineligible, and already done.

from cadybot import plays  # noqa: E402

LIVE = {
    "stage": "seed",
    "channels": {"count": 1, "all": [{"channel": "hermes"}]},
    "dead_channels": {"count": 1, "all": ["hermes"]},
    "unanswered_questions": [],
    "members": {"humans": 1, "never_posted": {"count": 0}, "gone_quiet": {"count": 1}},
    "activity": {"messages_30d": 0, "days_since_owner_posted": 88.7},
}
_eligible = [p.id for p in plays.eligible(LIVE)]
check("55 the play that produced the bad advice is not offered",
      "merge_channels" not in _eligible, _eligible)
check("55b and it is withheld as already done, not merely ineligible",
      plays.BY_ID["merge_channels"].status(LIVE) == "already true",
      plays.BY_ID["merge_channels"].status(LIVE))
check("55c answering an unanswered question is withheld when there are none",
      "answer_unanswered" not in _eligible)
check("55d but the plays that do fit are offered",
      {"post_the_work", "do_nothing_to_the_server"} <= set(_eligible), _eligible)

FOUR_DEAD = dict(LIVE, channels={"count": 5}, dead_channels={"count": 4})
check("56 with five channels and four dead, merging is offered",
      "merge_channels" in [p.id for p in plays.eligible(FOUR_DEAD)])

check("57 stage gates the catalogue",
      all("engagement_mechanics" != p.id for p in plays.eligible(LIVE))
      and "engagement_mechanics" in [p.id for p in plays.for_stage("growing")])

check("58 'none' is always choosable", "none" in plays.choices(LIVE))
check("58b the enum is exactly the eligible set plus none",
      plays.choices(LIVE) == _eligible + ["none"])

_M = advisor._brief_model_for(LIVE)
_base = dict(evidence="e", reasoning="r", would_change_my_mind="w", play_fails_when="f",
             headline="h", action="a", metric="none", direction="unchanged",
             horizon_days=7, guardrail_metric="none")
try:
    _M(recommendations=[dict(_base, play="merge_channels")], headline="x")
    _ok59 = False
except ValidationError:
    _ok59 = True
check("59 an ineligible play cannot be decoded at all", _ok59)
check("59b an eligible one can",
      _M(recommendations=[dict(_base, play="post_the_work")], headline="x") is not None)
_props = list(_M.model_json_schema()["$defs"]["EligibleRecommendation"]["properties"])
check("59c play is committed to before the free text is written",
      _props.index("play") < _props.index("action"), _props)
check("59d and the enum reaches the schema the grammar is built from",
      _M.model_json_schema()["$defs"]["EligibleRecommendation"]["properties"]["play"].get("enum")
      == plays.choices(LIVE))

check("60 plays.py stays model-free",
      not (imports((ROOT / "cadybot" / "plays.py").read_text()) & {"advisor", "llm"}))

# 61: the finding is written at provocation time and delivered on a later tick,
# so it has to survive the gap. It did not: Provocation.finding was computed and
# never stored, and the deferred path rebuilt a Provocation without it.
gid = fresh(900900)
db.connect().execute(
    "INSERT INTO journal (guild_id, kind, provoked_by, about_ref, started_at, "
    "self_prompt, outcome, to_founder, note_to_self, watch_metric, unverified, "
    "wanted_telling, finding) VALUES (?,?,?,?,?,?, 'thought', ?, ?, ?, '[]', 1, ?)",
    (gid, "backlog", ago(hours=3), None, ago(hours=3), "sp",
     "Answer them, even now.", "", "activity.messages_7d",
     "**someone** wrote in #hermes and nobody replied."))
held = agenda.unsaid(gid)
check("61 a stored finding survives to delivery", held is not None and held["finding"])
check("61b and is rendered above the prose",
      held is not None
      and advisor.render_stored(held).index("nobody replied")
        < advisor.render_stored(held).index("Answer them"))

# 61c: a later generic thought must not bury one that actually found something.
db.connect().execute(
    "INSERT INTO journal (guild_id, kind, provoked_by, about_ref, started_at, "
    "self_prompt, outcome, to_founder, note_to_self, watch_metric, unverified, "
    "wanted_telling, finding) VALUES (?,?,?,?,?,?, 'thought', ?, ?, ?, '[]', 1, NULL)",
    (gid, "verdict", ago(hours=1), "R-9", ago(hours=1), "sp",
     "Post an artifact this week.", "", "activity.messages_7d"))
check("61c a finding outranks a newer generic thought",
      (agenda.unsaid(gid) or {})["kind"] == "backlog")

# 62: the date exemption was stripping any d/d or d:dd pair, which hid two
# invented figures at a time — a false-positive fix creating false negatives in
# the check that matters more.
check("62 a real date still passes",
      not advisor.verify_evidence(_snap52, "nobody answered on 2026-05-25 at 17:46"))
check("62b but a ratio is not a date",
      advisor.verify_evidence(_snap52, "engagement is 3/4 of last month") == ["3", "4"])

# 63: a context thought can never be surfaced, so holding one in the single
# replay slot blocks every real finding behind it forever.
gid = fresh(900910)
for kind, finding in (("context", None), ("backlog", "**someone** was ignored.")):
    db.connect().execute(
        "INSERT INTO journal (guild_id, kind, provoked_by, about_ref, started_at, "
        "self_prompt, outcome, to_founder, note_to_self, watch_metric, unverified, "
        "wanted_telling, finding) VALUES (?,?,?,?,?,?, 'thought', ?, '', ?, '[]', 1, ?)",
        (gid, kind, ago(hours=2 if kind == "context" else 3), None,
         ago(hours=1 if kind == "context" else 3), "sp", "text",
         "activity.messages_7d", finding))
held = agenda.unsaid(gid)
check("63 a thought that can never be said does not block the queue",
      held is not None and held["kind"] == "backlog",
      held["kind"] if held else None)

# ------------------------------------------------------------------- report

print()
print("%d passed, %d failed" % (len(PASSED), len(FAILED)))
for name in FAILED:
    print("  failed: %s" % name)

for suffix in ("", "-wal", "-shm"):
    path = pathlib.Path(str(DB) + suffix)
    if path.exists():
        path.unlink()

sys.exit(1 if FAILED else 0)
