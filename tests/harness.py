"""Scenario harness.

Builds synthetic servers in a throwaway database and checks that cadybot reads
them correctly, then optionally that its advice is sane. Two tiers on purpose:

  tier 1  snapshot correctness — deterministic, fast, runs every scenario
  tier 2  advice sanity — needs a model, slow, runs a chosen subset

Run:
    .venv/bin/python tests/harness.py              # snapshot checks only
    .venv/bin/python tests/harness.py --advice     # also exercise the model
    .venv/bin/python tests/harness.py --only raid  # one scenario
"""

import argparse
import os
import pathlib
import random
import sys
import discord
import time
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "harness.db"
os.environ["CADYBOT_DB"] = "harness.db"
os.environ.pop("GUILD_ID", None)
os.environ.pop("OWNER_ID", None)

from cadybot import advisor, db, room, snapshot  # noqa: E402

NOW = datetime.now(timezone.utc)
random.seed(11)


def ts(days=0, hours=0):
    return (NOW - timedelta(days=days, hours=hours)).isoformat()


class Build:
    """Minimal fluent builder for a synthetic server."""

    def __init__(self, guild_id: int, name: str):
        self.g = guild_id
        self._mid = guild_id * 1000
        db.upsert_guild(guild_id, name)
        db.set_setting(guild_id, room.CHANNEL_KEY, str(guild_id + 1))

    def owner(self, uid: int):
        db.set_setting(self.g, room.OWNER_KEY, str(uid))
        return self

    def member(self, uid, name=None, bot=False, joined_days_ago=30, left=False):
        db.upsert_member(
            self.g, uid, name or ("u%d" % uid), name or ("u%d" % uid), bot, ts(joined_days_ago)
        )
        if left:
            db.mark_left(self.g, uid)
        return self

    def members(self, count, start=1, bot=False, joined_days_ago=30):
        for u in range(start, start + count):
            self.member(u, "u%d" % u, bot, joined_days_ago)
        return self

    def channel(self, cid, name, kind="TextChannel"):
        db.upsert_channel(self.g, cid, name, kind, None, ts(90))
        return self

    def msg(self, uid, cid, days_ago, text="hello", reply_to=None):
        self._mid += 1
        db.upsert_message(
            {
                "guild_id": self.g,
                "channel_id": cid,
                "message_id": self._mid,
                "author_id": uid,
                "created_at": ts(days_ago, random.randint(0, 23)),
                "content": text,
                "reply_to_id": reply_to,
                "attachments": 0,
                "reactions": 0,
            }
        )
        return self._mid

    def chatter(self, uids, cids, count, within_days=30):
        for _ in range(count):
            self.msg(random.choice(uids), random.choice(cids), random.randint(0, within_days))
        return self

    def join(self, uid, code=None):
        db.record_member_event(self.g, uid, "join", code)
        if code:
            db.attribute_invite(self.g, uid, code)
        return self

    def leave(self, uid):
        db.record_member_event(self.g, uid, "leave")
        return self

    def voice(self, uid, cid=9999):
        db.open_voice(self.g, cid, uid)
        db.close_voice(self.g, uid)
        return self

    def mention(self, message_id, author, targets):
        db.add_mentions(self.g, message_id, author, targets, ts(0))
        return self

    def react(self, message_id, uid, emoji="thumbsup"):
        db.add_reaction(self.g, message_id, uid, emoji)
        return self

    def thread(self, cid, name, parent, archived=0, messages=0, auto_archive=1440):
        db.upsert_channel(
            self.g, cid, name, "Thread", parent, ts(20), archived=archived,
            auto_archive_duration=auto_archive, thread_message_count=messages,
            parent_kind="TextChannel",
        )
        return self

    def audit(self, entry_id, action, target, days_ago=1):
        db.record_audit_event(self.g, entry_id, int(action.value), 1, target, ts(days_ago))
        return self

    def facts(self, **kw):
        db.upsert_guild_facts(self.g, kw)
        return self


SCENARIOS = []


def scenario(key, description, advice=False):
    def wrap(fn):
        SCENARIOS.append({"key": key, "description": description, "fn": fn, "advice": advice})
        return fn

    return wrap


def dig(obj, path):
    for part in path.split("."):
        obj = obj[part]
    return obj


# --- the scenarios ---------------------------------------------------------


@scenario("brand_new", "Installed today. One member, zero messages, no history.", advice=True)
def brand_new(b):
    b.member(1, "founder").owner(1).channel(10, "general")
    return {
        "stage": "seed",
        "members.humans": 1,
        "activity.messages_30d": 0,
        "activity.days_since_owner_posted": None,
        "unanswered_questions": lambda v: v == [],
        "members.never_posted.count": 1,
    }


@scenario("seven_friends", "The real situation: 7 friends, thin activity.", advice=True)
def seven_friends(b):
    b.members(7).owner(1).channel(10, "general").channel(11, "showcase")
    for d in range(9, 18):
        b.msg(1, 10, d, "shipped something")
    b.msg(2, 10, 15, "cool")
    b.msg(3, 11, 2, "does it do threaded holes? mine came out fused")
    return {
        "stage": "seed",
        "members.humans": 7,
        "unanswered_questions": lambda v: len(v) == 1 and "threaded" in v[0]["text"],
        "activity.days_since_owner_posted": lambda v: v >= 9,
        "members.never_posted.count": 4,
    }


@scenario("founder_silent", "Healthy-ish server the founder abandoned 40 days ago.", advice=True)
def founder_silent(b):
    b.members(40).owner(1).channel(10, "general")
    b.msg(1, 10, 40, "last thing i posted")
    b.chatter(list(range(2, 40)), [10], 200, within_days=25)
    return {
        "stage": "sprout",
        "activity.days_since_owner_posted": lambda v: v >= 40,
        "activity.owner_share_of_messages_7d": lambda v: v in (0, 0.0),
    }


@scenario("one_power_user", "One member produces almost everything. Key-person risk.", advice=True)
def one_power_user(b):
    b.members(120).owner(1).channel(10, "general")
    for _ in range(400):
        b.msg(2, 10, random.randint(0, 29))
    b.chatter(list(range(3, 30)), [10], 40)
    return {
        "stage": "growing",
        "top_posters_30d": lambda v: v[0]["messages_30d"] > 5 * v[1]["messages_30d"],
    }


@scenario("all_bots", "Twenty members, eighteen of them bots. Activity is fake.")
def all_bots(b):
    b.member(1, "founder").owner(1).channel(10, "general")
    b.member(2, "human2")
    for u in range(3, 21):
        b.member(u, "bot%d" % u, bot=True)
    for _ in range(300):
        b.msg(random.randint(3, 20), 10, random.randint(0, 29), "bot noise")
    b.msg(1, 10, 5, "anyone here")
    return {
        "members.humans": 2,
        "members.bots": 18,
        # Bot chatter must not count as activity — this is the whole point.
        "activity.messages_30d": 1,
    }


@scenario("dead_channels", "Thirty channels, two alive. Attention is fragmented.", advice=True)
def dead_channels(b):
    b.members(60).owner(1)
    for c in range(10, 40):
        b.channel(c, "chan-%d" % c)
    b.chatter(list(range(1, 60)), [10, 11], 300)
    return {
        "channels.count": 30,
        "dead_channels.count": 28,
    }


@scenario("exodus", "Sixty people left in a month after something went wrong.")
def exodus(b):
    b.members(100).owner(1).channel(10, "general")
    b.chatter(list(range(1, 40)), [10], 150)
    for u in range(41, 101):
        b.leave(u)
        db.mark_left(b.g, u)
    for u in range(101, 106):
        b.member(u)
        b.join(u)
    return {
        "membership_flow_30d.leaves": 60,
        "membership_flow_30d.joins": 5,
        "members.humans": lambda v: v == 45,
    }


@scenario("raid", "Two hundred accounts joined in a day and said nothing.")
def raid(b):
    b.members(210).owner(1).channel(10, "general")
    b.chatter([1, 2, 3], [10], 30)
    for u in range(11, 211):
        b.join(u)
    return {
        "membership_flow_30d.joins": 200,
        # The tell: joins spike, activation is ~zero.
        "members.never_posted_pct": lambda v: v > 90,
        "stage": "growing",
    }


@scenario("healthy_growing", "250 members, spread-out activity, replies happening.", advice=True)
def healthy_growing(b):
    b.members(250).owner(1)
    for c in range(10, 16):
        b.channel(c, "chan-%d" % c)
    b.chatter(list(range(1, 200)), list(range(10, 16)), 1200)
    for u in range(200, 230):
        b.join(u, code="abc")
    return {
        "stage": "growing",
        "activity.unique_posters_7d": lambda v: v > 20,
        "invite_attribution.count": lambda v: v >= 1,
    }


@scenario("exploding", "5,000 members. Everything must stay inside a context window.", advice=True)
def exploding(b):
    b.members(5000).owner(1)
    for c in range(10, 60):
        b.channel(c, "chan-%d" % c)
    talkers = list(range(1, 900))
    b.chatter(talkers, list(range(10, 60)), 8000, within_days=60)
    return {
        "stage": "community",
        "members.humans": 5000,
        "members.gone_quiet.count": lambda v: v > 100,
        "members.gone_quiet": lambda v: "sample" in v and len(v["sample"]) <= 12,
    }


@scenario("unanswered_pileup", "Six member questions nobody answered.", advice=True)
def unanswered_pileup(b):
    b.members(30).owner(1)
    for c in (10, 11, 12, 13, 14, 15):
        b.channel(c, "chan-%d" % c)
        b.msg(c - 8, c, 2, "how do i make this fit? anyone?")
    return {
        "unanswered_questions": lambda v: len(v) == 6,
    }


@scenario("seasonal_lull", "Was busy, went quiet three weeks ago.")
def seasonal_lull(b):
    b.members(150).owner(1).channel(10, "general")
    for _ in range(600):
        b.msg(random.randint(1, 150), 10, random.randint(21, 60))
    return {
        "activity.messages_7d": 0,
        "activity.messages_30d": lambda v: v > 0,
        "members.gone_quiet.count": lambda v: v > 50,
    }


@scenario("voice_heavy", "Nobody types, everybody talks.")
def voice_heavy(b):
    b.members(40).owner(1).channel(10, "general")
    b.msg(1, 10, 3, "vc?")
    for u in range(1, 35):
        b.voice(u)
    return {
        "voice_30d.unique_participants": 34,
        "activity.messages_30d": 1,
    }


@scenario("owner_talks_to_self", "Founder posts constantly, nobody replies.", advice=True)
def owner_talks_to_self(b):
    b.members(25).owner(1).channel(10, "general")
    for d in range(0, 25):
        b.msg(1, 10, d, "update %d" % d)
    return {
        "activity.owner_share_of_messages_7d": lambda v: v == 1.0,
        "members.never_posted.count": 24,
    }


@scenario("no_history_yet", "Members exist but backfill has not run.")
def no_history_yet(b):
    b.members(80).owner(1).channel(10, "general")
    return {
        "activity.messages_30d": 0,
        "members.never_posted.count": 80,
        "members.gone_quiet.count": 0,
    }


@scenario("moderation_wave", "Fifty departures — but forty were bans, not churn.", advice=True)
def moderation_wave(b):
    """The distinction that flips the advice.

    Fifty people leaving looks identical to fifty people being removed unless
    the audit log is read. One means the community is failing; the other means
    it is being defended. Without moderation_30d an advisor confidently
    diagnoses churn and prescribes retention work for a raid cleanup.
    """
    b.members(120).owner(1).channel(10, "general")
    b.chatter(list(range(1, 60)), [10], 200)
    for i, u in enumerate(range(61, 111)):
        b.leave(u)
        db.mark_left(b.g, u)
        if i < 40:
            b.audit(1000 + i, discord.AuditLogAction.ban, u, days_ago=2)
    return {
        "membership_flow_30d.leaves": 50,
        "moderation_30d.bans": 40,
        "moderation_30d.kicks": 0,
    }


@scenario("thread_culture", "A forum-shaped server where threads die young.")
def thread_culture(b):
    b.members(80).owner(1).channel(10, "help")
    for i in range(12):
        b.thread(200 + i, "thread-%d" % i, 10, archived=1 if i < 10 else 0, messages=2 + i % 3)
    b.chatter(list(range(1, 60)), [10], 120)
    return {
        # The bug this pins: threads and categories were counted as channels, so
        # a server with one text channel and twelve threads reported 13 and got
        # told to prune them. Threads are conversations, not structure.
        "channels.count": 1,
        "dead_channels.count": lambda v: v <= 1,
        "threads.total": 12,
        "threads.archived": 10,
        "threads.still_active": 2,
        "threads.median_messages_in_archived": lambda v: v is not None,
        # No archiver_id column exists, so this must stay honestly unknown
        # rather than guessing that every archive was a timeout.
        "threads.archived_by_timeout.value": None,
    }


@scenario("owner_shouting", "Only the founder ever @-mentions anyone.")
def owner_shouting(b):
    b.members(30).owner(1).channel(10, "general")
    for d in range(0, 12):
        mid = b.msg(1, 10, d, "hey @u%d take a look" % (d + 2))
        b.mention(mid, 1, [d + 2])
    mid = b.msg(5, 10, 3, "nice")
    b.react(mid, 1)
    return {
        "structure.mentions_30d": 12,
        # Nobody but the founder pulls anyone in — a community of one hub.
        "structure.mentioned_only_by_owner.count": lambda v: v >= 10,
        "structure.reciprocity": None,
    }


@scenario("config_unreadable", "Permissions missing: unknown must not read as zero.")
def config_unreadable(b):
    b.members(40).owner(1).channel(10, "general")
    b.chatter(list(range(1, 40)), [10], 60)
    b.facts(onboarding_enabled=1, onboarding_readable=1)
    return {
        # Readable fact comes back bare; unreadable ones carry a reason. The
        # whole NULL-means-unreadable design fails silently if these collapse.
        "server_config.onboarding_enabled": 1,
        "server_config.onboarding_mode": lambda v: isinstance(v, dict) and v["value"] is None and v["reason"],
    }


# --- checks ----------------------------------------------------------------

SEED_FORBIDDEN = (
    "event", "tournament", "contest", "leaderboard", "xp ", "office hours",
    "ama", "giveaway", "weekly ritual", "competition",
)
ALWAYS_FORBIDDEN = ("dm them all", "mass dm", "bulk dm", "buy members", "engagement pod")


def check_advice(key, stage, brief):
    """Structural checks on a Brief. Cheap, and catches the failures that matter."""
    problems = []
    blob = " ".join(
        [brief.headline, brief.dont or ""]
        + [r.headline + " " + r.action for r in brief.recommendations]
    ).lower()

    if not 1 <= len(brief.recommendations) <= 3:
        problems.append("returned %d recommendations (want 1-3)" % len(brief.recommendations))

    for rec in brief.recommendations:
        if not rec.evidence.strip():
            problems.append("recommendation %r cites no evidence" % rec.headline[:40])
        if not rec.metric.strip():
            problems.append("recommendation %r names no metric" % rec.headline[:40])

    for phrase in ALWAYS_FORBIDDEN:
        if phrase in blob:
            problems.append("suggested %r, which is banned at every stage" % phrase)

    if stage == "seed":
        for phrase in SEED_FORBIDDEN:
            if phrase in blob:
                problems.append("suggested %r at seed stage — stage gate breached" % phrase.strip())

    return problems


def run(only=None, with_advice=False):
    if DB.exists():
        DB.unlink()

    passed = failed = 0
    advice_runs = []

    for i, sc in enumerate(SCENARIOS):
        if only and only not in sc["key"]:
            continue
        gid = 100000 + i * 10
        expectations = sc["fn"](Build(gid, sc["key"]))
        snap = snapshot.build(gid)

        problems = []
        for path, expected in expectations.items():
            try:
                actual = dig(snap, path)
            except (KeyError, TypeError, IndexError) as exc:
                problems.append("%s missing (%s)" % (path, exc))
                continue
            ok = expected(actual) if callable(expected) else actual == expected
            if not ok:
                shown = expected if not callable(expected) else "<predicate>"
                problems.append("%s = %r, wanted %r" % (path, actual, shown))

        if problems:
            failed += 1
            print("FAIL  %-20s %s" % (sc["key"], sc["description"]))
            for p in problems:
                print("        %s" % p)
        else:
            passed += 1
            print("ok    %-20s stage=%-9s humans=%-5s msgs30=%s"
                  % (sc["key"], snap["stage"], snap["members"]["humans"],
                     snap["activity"]["messages_30d"]))

        if with_advice and sc["advice"]:
            advice_runs.append((sc, gid, snap))

    print("\n%d passed, %d failed" % (passed, failed))

    if advice_runs:
        print("\n--- advice sanity (%d scenarios, this is slow) ---" % len(advice_runs))
        for sc, gid, snap in advice_runs:
            t = time.time()
            try:
                brief = advisor.brief(snap, gid)
            except Exception as exc:
                failed += 1
                print("FAIL  %-20s advisor raised: %s" % (sc["key"], exc))
                continue
            problems = check_advice(sc["key"], snap["stage"], brief)
            mark = "FAIL" if problems else "ok  "
            print("%s  %-20s %2d recs, %.0fs — %s"
                  % (mark, sc["key"], len(brief.recommendations), time.time() - t,
                     brief.headline[:80]))
            for p in problems:
                print("        %s" % p)
            if problems:
                failed += 1
            else:
                passed += 1

    if DB.exists():
        DB.unlink()
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--advice", action="store_true", help="also run the model (slow)")
    ap.add_argument("--only", help="substring match on scenario key")
    args = ap.parse_args()
    sys.exit(run(only=args.only, with_advice=args.advice))
