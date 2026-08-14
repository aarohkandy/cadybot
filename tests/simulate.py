"""Run the desk against a virtual clock, for months, and count what it says.

Every cadence decision until now was an argument. This makes it a measurement:
the real agenda, probe, plays and thinking modules, driven over simulated weeks
against servers of different shapes, with the model stubbed so only the
scheduling logic is under test.

The question it exists to answer is the one the founder actually asked — *why
does it not ping me more?* — and the honest way to answer that is to watch a
month go by rather than to reason about the constants.

    .venv/bin/python tests/simulate.py            # all scenarios
    .venv/bin/python tests/simulate.py dead       # just one

Nothing here touches the live database or the network.
"""

import datetime as dt
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "simulate.db"
os.environ["CADYBOT_DB"] = "simulate.db"
os.environ.pop("GUILD_ID", None)
os.environ.pop("OWNER_ID", None)

from cadybot import advisor, agenda, config, db, ledger, scorecard, snapshot  # noqa: E402

# --- the virtual clock ------------------------------------------------------
#
# agenda and probe read "now" through db.now/days_ago/hours_ago and
# ledger.today/day_offset. Swapping those four is enough to move the whole desk
# through time without touching a line of the code under test.

CLOCK = {"t": dt.datetime(2026, 3, 1, 9, 0, tzinfo=dt.timezone.utc)}


def _iso(d):
    return d.isoformat()


db.now = lambda: _iso(CLOCK["t"])
db.days_ago = lambda n: _iso(CLOCK["t"] - dt.timedelta(days=n))
db.hours_ago = lambda n: _iso(CLOCK["t"] - dt.timedelta(hours=n))
ledger.today = lambda: CLOCK["t"].date().isoformat()
ledger.day_offset = lambda n: (CLOCK["t"].date() - dt.timedelta(days=n)).isoformat()


class Stub:
    """A model that always has something to say, so the gates are what is tested.

    Deliberately the most talkative model possible: it lifts the veto every
    time, cites nothing unverifiable, and never fails. Anything that keeps
    cadybot quiet in this simulation is the scheduling, not the model.
    """

    def __init__(self):
        self.calls = 0

    def __call__(self, *a, **k):
        self.calls += 1
        return advisor.Reflection(
            restated="what does this mean?",
            reasoning="working",
            evidence="messages_7d is what it is",
            note_to_self="keep an eye on whether anyone replies",
            watch_metric="activity.messages_7d",
            worth_telling_founder=True,
            to_founder="Somebody is waiting on you. Go and answer them.",
        )


# --- world building ---------------------------------------------------------


def reset(gid):
    conn = db.connect()
    for table in db.GUILD_TABLES:
        conn.execute("DELETE FROM %s WHERE guild_id=?" % table, (gid,))
    conn.execute("INSERT INTO guilds (guild_id, name, first_seen) VALUES (?,?,?)",
                 (gid, "sim", db.days_ago(120)))
    conn.execute("INSERT OR REPLACE INTO channels (guild_id, channel_id, name, kind) "
                 "VALUES (?,?,?,'TextChannel')", (gid, 1, "general"))


def person(gid, uid, is_bot=0):
    db.connect().execute(
        "INSERT OR REPLACE INTO members (guild_id, user_id, username, display_name, "
        "is_bot, first_seen) VALUES (?,?,?,?,?,?)",
        (gid, uid, "u%d" % uid, "u%d" % uid, is_bot, db.days_ago(100)))


_MID = {"n": 1}


def says(gid, uid, when=None):
    _MID["n"] += 1
    db.connect().execute(
        "INSERT INTO messages (guild_id, channel_id, message_id, author_id, "
        "created_at, content, type) VALUES (?,1,?,?,?,?,0)",
        (gid, _MID["n"], uid, when or db.now(), "a message with some words in it"))


def joins(gid, uid):
    person(gid, uid)
    db.connect().execute(
        "INSERT INTO member_events (guild_id, user_id, event, at) VALUES (?,?,'join',?)",
        (gid, uid, db.now()))


# --- the loop ---------------------------------------------------------------


def run(gid, days, world, label):
    """Tick the desk on its real schedule for `days`, letting `world` act daily."""
    stub = Stub()
    advisor.llm.generate = stub
    said, thoughts, kinds = [], 0, {}
    tick_every = config.THINK_INTERVAL_HOURS
    start = CLOCK["t"]

    for day in range(days):
        for hour in range(24):
            CLOCK["t"] = start + dt.timedelta(days=day, hours=hour)
            if hour == 9:
                world(gid, day)
            # the hourly ledger close, as listener.hourly_facts does it
            try:
                ledger.record_day(gid, snapshot.build(gid))
            except Exception:
                pass
            if hour % tick_every:
                continue

            # one desk pass, inlined from thinking.think so the simulation does
            # not need an event loop or a Discord client
            agenda.reap_stale(gid)
            waiting = agenda.unsaid(gid)
            if waiting is not None:
                held = agenda.Provocation(waiting["kind"], waiting["provoked_by"],
                                          "", waiting["about_ref"])
                ok, _ = agenda.may_surface(gid, held)
                if ok:
                    agenda.mark_surfaced(waiting["id"])
                    db.record_delivery(gid, "thought", 200)
                    said.append((CLOCK["t"], waiting["kind"]))
                    continue
            if not agenda.affordable(gid):
                continue
            prov = agenda.next_provocation(gid, None)
            if prov is None:
                continue
            jid = agenda.open_attempt(gid, prov)
            if jid is None:
                continue
            thoughts += 1
            kinds[prov.kind] = kinds.get(prov.kind, 0) + 1
            result = stub()
            agenda.record_result(jid, result, [], "stub")
            ok, _ = agenda.may_surface(gid, prov)
            if ok and result.worth_telling_founder:
                agenda.mark_surfaced(jid)
                db.record_delivery(gid, "thought", 200)
                said.append((CLOCK["t"], prov.kind))

    weeks = max(days / 7.0, 1e-9)
    print("  %-22s %2d thoughts, %2d messages  (%.1f msgs/week)  %s"
          % (label, thoughts, len(said), len(said) / weeks,
             ", ".join("%s×%d" % (k, v) for k, v in sorted(kinds.items())) or "-"))
    return len(said), thoughts


# --- scenarios --------------------------------------------------------------


def dead(gid, day):
    pass


def wakes_up(gid, day):
    if day == 10:
        says(gid, 1)


def one_person_asking(gid, day):
    """A real small server: somebody asks something every few days."""
    if day % 4 == 0:
        says(gid, 2)


def healthy(gid, day):
    """Conversation: several people, and things get answered."""
    for uid in (1, 2, 3):
        says(gid, uid)
    if day % 9 == 0:
        joins(gid, 100 + day)


SCENARIOS = {
    "dead": ("a dead server, 60 days", 60, dead, 1),
    "wakes": ("silent, then one message", 40, wakes_up, 1),
    "trickle": ("one asker every 4 days", 60, one_person_asking, 2),
    "healthy": ("3 people talking daily", 30, healthy, 3),
}


def main():
    db.connect()
    scorecard.ensure_schema()
    want = sys.argv[1:] or list(SCENARIOS)
    print("cadence: tick %dh | %d thoughts/day | %d msgs/week | %dh gap | %s-%s UTC"
          % (config.THINK_INTERVAL_HOURS, config.THINK_CALLS_PER_DAY,
             config.SURFACE_MAX_PER_WEEK, config.SURFACE_MIN_GAP_HOURS,
             *config.SURFACE_WINDOW_UTC))
    print()
    gid = 5000
    for name in want:
        if name not in SCENARIOS:
            continue
        label, days, world, humans = SCENARIOS[name]
        gid += 1
        CLOCK["t"] = dt.datetime(2026, 3, 1, 9, 0, tzinfo=dt.timezone.utc)
        reset(gid)
        for uid in range(1, humans + 1):
            person(gid, uid)
        # history, so `backlog` has something to find, as on any real install
        for i in range(12):
            says(gid, 1, when=db.days_ago(70 + i))
        agenda.installed_at(gid)
        run(gid, days, world, label)

    for suffix in ("", "-wal", "-shm"):
        p = pathlib.Path(str(DB) + suffix)
        if p.exists():
            p.unlink()


main()
