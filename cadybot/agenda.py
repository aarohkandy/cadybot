"""What is worth thinking about, decided by code rather than by a model.

This is the module that makes cadybot self-prompting. Four times a day it asks
"has anything happened that I have not thought about yet?", and almost always
the answer is no and nothing runs.

The whole design rests on one rule:

    **A provocation's timestamp is read from a row in the database or from a
    file on disk. Never from the clock.**

That sentence is the difference between an agent that thinks when there is
something to think about and one that thinks because time passed. Three separate
designs for this feature were drafted and every one of them, in a different
place, ended up watching a number that changes on its own —
`activity.days_since_owner_posted` rises by 1.0 every day precisely *because*
nobody posts, so a detector pointed at it fires forever on a dead server, learns
nothing each time, and bills for the privilege. That is the failure this file
exists to make structurally impossible.

Two consequences fall out of the rule, both of which do real work:

- **Silence is not a policy, it is the absence of input.** A dead server
  produces no rows, so it produces no provocations. The bot cannot drift into
  chattiness any more than grep can match a line in an empty file. This is what
  separates it from a scheduled agent that must decide, every cycle, to stay
  quiet — and eventually decides wrong.
- **Deduplication is free and works across processes.** Because `provoked_by` is
  copied from stored data rather than generated, the listener and a stray `cadybot
  think` compute byte-identical values, and `UNIQUE (guild_id, kind, provoked_by)`
  makes the second one lose. A provocation exists once and is consumed once, so
  there is no backlog to drain and no cooldown to tune.

Model-free, and bound by the same ban as scorecard.py and ledger.py: nothing
here may import advisor or llm. Deciding what deserves a thought is exactly the
kind of judgement a model would make generously, and generosity is the failure
mode. `advisor` may import this module; this module may never import `advisor`.
"""

import dataclasses
import datetime
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from . import config, db, ledger, probe, scorecard, snapshot

# When this guild first grew an agenda. Everything with an earlier timestamp is
# history rather than news, which is what stops installation day from firing
# every generator at once over a backlog the founder has already lived through.
INSTALLED_KEY = "agenda_installed_at"

# Set by /quiet. Suppresses speaking, never thinking — the record keeps
# accumulating and `/notes` keeps showing it.
QUIET_KEY = "agenda_quiet_until"

# A tally of how often the desk has looked. Kept so "it is constantly thinking"
# is a number the founder can read rather than a claim — an idle pass leaves no
# journal row by design, so without this the work is invisible and indeed
# indistinguishable from a dead loop.
SCANS_KEY = "agenda_scans"

# Precedence. First match wins, and there is no scoring: a ranking formula over
# hand-tuned weights was tried in design and froze solid after one cycle, which
# is what ranking formulas over hand-tuned weights do.
KINDS = ("ignored", "backlog", "verdict", "life", "joined", "context",
         "drift", "digest")

# Which kinds may ever reach the founder unprompted. `context` is deliberately
# absent: reacting to the founder editing his own notes file is thinking worth
# recording and never an interruption worth making.
SURFACEABLE = ("ignored", "backlog", "verdict", "life", "joined", "drift",
               "digest")

# The one generator allowed to point at a timestamp older than the agenda.
# Everything else treats the past as history the founder has already lived
# through; `backlog` exists precisely because he has not — nothing had ever read
# it. It still fires once, because its timestamp is the newest historical
# message and that does not move.
LOOKS_BACK = ("backlog",)

# A join is only news if it broke a drought. On a 500-member server joins are
# weather; on a one-human server the first one in a fortnight is the single most
# decision-relevant thing that can happen.
JOIN_DROUGHT_DAYS = 14

# Joins closer together than this are one arrival, not several. Without it, a
# post getting shared produces a burst in which every join cancels the drought
# of the one after it, and the single most encouraging thing that can happen to
# a small server generates no thought at all.
JOIN_BURST_DAYS = 2

# How many times one provocation may be attempted. A model call that dies takes
# its provocation with it, which is right for a thought that failed on its
# merits and wrong for one that failed because ollama was restarting.
MAX_ATTEMPTS = 2

# A ceiling on any lookup run from the desk. unanswered_history carries a
# correlated NOT EXISTS over messages, which is quadratic in channel size; on a
# busy server an unbounded one could outlast the whole pass. probe.run's own
# deadline is optional and the desk was not passing one.
PROBE_DEADLINE_S = 20

# Must match advisor._NUMERAL, which is what verify_evidence tokenises with.
_NUMERAL = re.compile(r"-?\d+(?:\.\d+)?")


@dataclasses.dataclass
class Provocation:
    """Something that happened, and the question it raises.

    `self_prompt` is composed here, by code, from stored values. No model-authored
    text is ever stored as a question: a question is durable prompt input, and a
    number invented inside one becomes a fact cadybot believes next month.
    """

    kind: str
    provoked_by: str                    # from a stored row or a file mtime
    self_prompt: str
    about_ref: Optional[str] = None
    numbers: Tuple[str, ...] = ()       # figures this block injected, for verify_evidence
    # A sentence composed here, from SQL, stating the finding itself. Rendered
    # above whatever the model writes. Three runs of the same provocation on the
    # live server produced the finding once and generic advice twice, and two
    # larger local models did no better — one of them recommended recognising
    # the contributions of a bot. The fact is deterministic; only the prose
    # around it needs a model, so the fact stops depending on one.
    finding: Optional[str] = None


# --- installation ----------------------------------------------------------


def record_scan(guild_id: int, found: Optional[str]) -> Dict[str, Any]:
    """Note that the desk looked, and what it saw. One tiny write, no model.

    Resets daily so the count means "today" rather than "since install".
    """
    today = ledger.today()
    try:
        state = json.loads(db.get_setting(guild_id, SCANS_KEY) or "{}")
    except ValueError:
        state = {}
    if state.get("day") != today:
        state = {"day": today, "scans": 0, "found": 0}
    state["scans"] = int(state.get("scans", 0)) + 1
    if found:
        state["found"] = int(state.get("found", 0)) + 1
        state["last_found"] = found
    state["last_at"] = db.now()
    db.set_setting(guild_id, SCANS_KEY, json.dumps(state))
    return state


def scans_today(guild_id: int) -> Dict[str, Any]:
    try:
        state = json.loads(db.get_setting(guild_id, SCANS_KEY) or "{}")
    except ValueError:
        return {}
    return state if state.get("day") == ledger.today() else {}


def installed_at(guild_id: int, create: bool = True) -> Optional[str]:
    """When this guild's agenda opened, writing it on first ask.

    `create=False` looks without touching. The mark decides forever what counts
    as history rather than news, so writing it is the most consequential thing
    a first call can do — and `cadybot reflect`, documented as reading only, was
    doing exactly that on any guild the desk had not run for yet.
    """
    stamp = db.get_setting(guild_id, INSTALLED_KEY)
    if not stamp and create:
        stamp = db.now()
        db.set_setting(guild_id, INSTALLED_KEY, stamp)
    return stamp


# --- the five generators ---------------------------------------------------
#
# Each returns a Provocation or None. Each reads its timestamp out of storage.




def _from_digest(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """The day is over and nothing else got said. Say something anyway.

    Last in precedence, and the only generator that is not about a surprise.
    Simulating sixty days of the live server produced exactly one message, and a
    *healthy* server produced fewer than a struggling one, because everything
    getting answered means nothing is ever wrong. Purely event-driven turned out
    to be the wrong shape: a founder who hears from his advisor once in two
    months does not have an advisor.

    This is still not the clock. `provoked_by` is yesterday's ledger close — a
    row, written by listener.hourly_facts, that exists only if the collector
    actually ran. So a day cadybot was down produces no digest and no false
    "nothing happened", and the UNIQUE key gives one per day for free.

    It fires only when nothing else has been delivered that day, so on an active
    server the real findings crowd it out and it never doubles up.
    """
    day = ledger.day_offset(1)
    closed = db.one(
        "SELECT day FROM ledger WHERE guild_id=? AND day=? LIMIT 1", (guild_id, day)
    )
    if not closed:
        return None                      # the collector did not run; say nothing
    if db.deliveries_since(guild_id, db.hours_ago(20)):
        return None                      # he has already heard from us today

    moved = []
    for metric in ledger.LEDGER_METRICS:
        before = ledger.value_on(guild_id, ledger.day_offset(8), metric)
        after = ledger.value_on(guild_id, day, metric)
        if before is None or after is None or before == after:
            continue
        moved.append("%s %s -> %s" % (metric, _fmt(before), _fmt(after)))

    # A day where something moved is worth a daily note. A day where nothing
    # did is worth one every few days — simulating sixty days of the live server
    # with an unconditional digest produced fifty-nine consecutive messages
    # saying "nothing moved", which is the noise failure this whole design
    # exists to avoid, just wearing a schedule.
    if not moved and db.deliveries_since(
        guild_id, db.days_ago(config.DIGEST_QUIET_DAYS), "thought"
    ):
        return None

    open_row = scorecard.open_row(guild_id)
    if moved:
        finding = "Week on week: " + "; ".join(moved[:3]) + "."
    elif open_row:
        finding = ("Nothing moved this week. %s is still open — day %.0f of %d."
                   % (open_row["ref"], open_row["age_days"],
                      open_row["horizon_days"] or config.RECOMMENDATION_HORIZON_DAYS))
    else:
        finding = "Nothing moved this week, and nothing is outstanding."

    prompt = "\n".join([
        "# Where things stand",
        "",
        "Nothing in particular happened. This is the daily check-in, and the",
        "numbers below are the whole of what changed since a week ago:",
        "",
        "  " + ("\n  ".join(moved) if moved else "nothing measurable moved"),
        "",
        "# Your question",
        "",
        "One or two sentences. If something moved, say what you make of it. If",
        "nothing did — which is the normal case on a small server — say that",
        "plainly and say what you are watching for. Do not manufacture an",
        "insight and do not repeat advice he has already had; a short honest",
        "\"still quiet, here is the one thing I am waiting to see\" is exactly",
        "right and is what he is paying for.",
    ])
    return Provocation("digest", day + "T00:00:00+00:00", prompt, None, (), finding)


def _from_ignored(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """Somebody wrote something and nobody answered them.

    First in precedence, ahead of everything including the backlog, because it
    is the only provocation about a person who is currently being let down
    rather than a number that moved. It is also the one thing on this list the
    founder can fix in sixty seconds.

    Fires per message, so an active server produces these regularly and a dead
    one produces none — which is the honest way to be more frequent. The answer
    window is `config.RESPONSE_WINDOW_HOURS` (48), so a message only becomes a
    provocation once it is genuinely past being answered in the normal course,
    and `provoked_by` is that message's own timestamp.
    """
    cutoff = db.hours_ago(config.RESPONSE_WINDOW_HOURS)
    row = db.one(
        "SELECT m.created_at, m.content, c.name AS channel, "
        "       COALESCE(mem.display_name, mem.username, 'someone who has since left') AS who, "
        "       (mem.user_id IS NOT NULL) AS on_roster "
        "FROM messages m "
        "LEFT JOIN channels c ON c.guild_id=m.guild_id AND c.channel_id=m.channel_id "
        "LEFT JOIN members mem ON mem.guild_id=m.guild_id AND mem.user_id=m.author_id "
        "WHERE m.guild_id=? AND m.type IN (0, 19) AND COALESCE(mem.is_bot,0)=0 "
        "  AND m.created_at > ? AND m.created_at < ? "
        "  AND m.content IS NOT NULL AND LENGTH(m.content) > 3 "
        "  AND NOT EXISTS ("
        "     SELECT 1 FROM messages r "
        "     LEFT JOIN members rm ON rm.guild_id=r.guild_id AND rm.user_id=r.author_id "
        "     WHERE r.guild_id=m.guild_id AND r.channel_id=m.channel_id "
        "       AND r.author_id <> m.author_id AND r.type IN (0, 19) "
        "       AND COALESCE(rm.is_bot,0)=0 "
        "       AND julianday(r.created_at) > julianday(m.created_at) "
        "       AND julianday(r.created_at) <= julianday(m.created_at) + ?) "
        "ORDER BY m.created_at DESC LIMIT 1",
        (guild_id, since, cutoff, config.RESPONSE_WINDOW_HOURS / 24.0),
    )
    if not row:
        return None
    text, _n = probe._redact(row["content"])
    finding = ("**%s** asked something in #%s on %s and still has no reply."
               % (row["who"], row["channel"] or "?", row["created_at"][:10]))
    prompt = "\n".join([
        "# What happened",
        "",
        "%s wrote this in #%s on %s, and %d hours later nobody has replied:"
        % (row["who"], row["channel"] or "?", row["created_at"][:10],
           config.RESPONSE_WINDOW_HOURS),
        "",
        "  %s" % text[:400],
        "",
        "# Your question",
        "",
        "Tell him to answer this person, and say what you would answer. Not",
        "\"engage with your community\" — the actual reply, or the one question",
        "worth asking back. If the message needs no reply, say that plainly.",
    ])
    return Provocation("ignored", row["created_at"], prompt, None, (), finding)


def _from_backlog(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """The message history exists and has never been read.

    Every other provocation waits for something to happen. This one fires on
    something that already did: there are messages in this database that predate
    the agenda, and until probe.py existed no model could read a word of them.
    A corpus becoming readable for the first time is a real event, and the rows
    that make it real are the messages themselves.

    Fires exactly once per server, because `_already_thought` keys on
    provoked_by and the newest historical message does not move. It is the only
    provocation that looks backwards, and that is the point — a bot installed on
    a server with three months of history should not have to wait for the fourth
    month to have something to say.
    """
    row = db.one(
        "SELECT MAX(m.created_at) AS at, COUNT(*) AS n FROM messages m "
        "LEFT JOIN members mem ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id=? AND m.type IN (0, 19)",
        (guild_id,),
    )
    if not row or not row["at"] or (row["n"] or 0) < 10:
        return None
    if row["at"] >= since:
        return None                       # not a backlog; the live generators own it

    # Real lookups, run now, so the question carries evidence rather than a
    # suggestion to go and find some.
    blocks = []
    numbers: List[str] = []
    # unanswered_history first, and deliberately: an ignored message is one
    # named person who wanted something and did not get it, which is the most
    # actionable artefact a small server produces. Counts go after it, because a
    # model handed counts first writes about counts.
    for name, args in (("unanswered_history", {}), ("roster_authors", {}),
                       ("table_freshness", {})):
        finding = probe.run(guild_id, name, args, deadline_s=PROBE_DEADLINE_S)
        if finding.error:
            continue
        blocks.append("## %s\n%s" % (name, finding.body[:900]))
        numbers.extend(_numerals(*[v for v in _flat_numbers(finding.facts)]))

    # The headline fact, decided by SQL rather than by whoever is reading.
    ignored = probe.run(guild_id, "unanswered_history", {"limit": 4},
                        deadline_s=PROBE_DEADLINE_S)
    finding = None
    if ignored.rows and ignored.facts.get("ignored"):
        first = ignored.facts["ignored"][0]
        gone = " They are no longer in the server." if not first.get("on_roster") else ""
        finding = (
            "**%s** wrote in #%s on %s and nobody ever replied.%s"
            % (first["author"], first["channel"] or "?", (first["at"] or "")[:10], gone)
        )
        # The TOTAL, not the number the LIMIT happened to return. This line
        # went out to the founder reading "4 messages in this server's history
        # were never answered" on a server where the true answer is 17, in the
        # sentence this file calls the fact decided by SQL.
        total = ignored.facts.get("total") or ignored.rows
        if total > 1:
            finding += (" %d messages in this server's history were never "
                        "answered." % total)

    prompt = "\n".join([
        "# What happened",
        "",
        "Nothing just happened. This is the opposite: this server has %d messages"
        % row["n"],
        "of history, the newest from %s, and until now no model has ever been able"
        % row["at"][:10],
        "to read any of it. You can now. Below is what the records actually say.",
        "",
    ] + blocks + [
        "",
        "# Your question",
        "",
        "What is true about this server that the founder probably does not know?",
        "",
        "Rules for the answer, in order:",
        "1. If somebody asked for something and nobody answered them, that is the",
        "   answer. Say who, say when, say what they asked for, and say whether",
        "   they are still here. One ignored person outranks every statistic on",
        "   this page.",
        "2. Otherwise, if the records show something structurally surprising — a",
        "   bot wrote most of the messages, people wrote and left before anything",
        "   was recording, a whole class of event has never been collected — lead",
        "   with that.",
        "3. Only if neither holds, fall back on advice.",
        "",
        "Do not open with what he already knows. He knows the server is quiet. He",
        "knows he has not posted. Telling him again is worse than saying nothing.",
    ])
    return Provocation("backlog", row["at"], prompt, None,
                       tuple(dict.fromkeys(numbers)), finding)


def _from_verdict(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """A recommendation closed. The one moment cadybot has an oracle.

    This is the provocation that matters most, because it is the only one where
    the model is handed a fact about its own past judgement that it did not
    produce and cannot revise. Huang et al. (ICLR 2024) found intrinsic
    self-correction — a model reconsidering with no external signal — is net
    negative; a verdict computed by scorecard.py is precisely the external
    signal that makes reflection worth doing rather than harmful.
    """
    # The verdict columns are added by scorecard's own migration, not by
    # db.SCHEMA, so on a database where that has never run they do not exist and
    # this SELECT raises. agenda must not import scorecard — that would put the
    # grader one import from the model path — so ask the database instead.
    present = set(
        r[1] for r in db.connect().execute("PRAGMA table_info(recommendations)")
    )
    if not {"verdict", "verdict_at"} <= present:
        return None
    row = db.one(
        "SELECT id, action, prediction, verdict, verdict_at, verdict_current, "
        "       verdict_pvalue, baseline, threshold, metric, created_at "
        "FROM recommendations "
        "WHERE guild_id=? AND verdict IS NOT NULL AND verdict_at > ? "
        "ORDER BY verdict_at DESC LIMIT 1",
        (guild_id, since),
    )
    if not row:
        return None

    numbers = _numerals(row["baseline"], row["threshold"], row["verdict_current"])
    prompt = "\n".join(
        [
            "# What happened",
            "",
            "%s closed on %s with the verdict `%s`."
            % (scorecard.ref(row["id"]), row["verdict_at"], row["verdict"]),
            "",
            "  action (your words, stored %s): %r" % (row["created_at"], row["action"]),
            "  prediction (composed by code at issue time): %r" % (row["prediction"] or "",),
            "  metric %s — baseline %s, threshold %s, reading at close %s"
            % (
                row["metric"],
                _fmt(row["baseline"]),
                _fmt(row["threshold"]),
                _fmt(row["verdict_current"]),
            ),
            "",
            "This verdict was computed from the numbers by code you did not run and",
            "cannot revise. It is a fact. There is no field here for re-grading it.",
            "",
            "# Your question",
            "",
            "What did you believe about this server when that was written that the",
            "outcome now says was wrong? Say what you would want to know before",
            "writing anything like it again. \"Nothing — the reading was right and the",
            "founder did not act\" is a complete and frequently correct answer.",
        ]
    )
    # The verdict, stated by the code that computed it. Left to the model, it
    # gets relabelled: an `inconclusive` grade came back to the founder as "the
    # prediction failed... even after your outreach", which invents both a
    # verdict and an action he was never recorded as taking. This is the same
    # reason render_scorecard prints _VERDICT_LABEL rather than letting a model
    # narrate an outcome.
    finding = "%s: %s — %s." % (
        scorecard.ref(row["id"]),
        {"worked": "moved as predicted", "failed": "did not move",
         "harmful": "guardrail broke", "revoked": "moved, then fell back",
         "inconclusive": "no separable change",
         "not_attempted": "no sign it was tried",
         "unmeasurable": "not measurable"}.get(row["verdict"], row["verdict"]),
        "%s was %s at issue and is %s now"
        % (row["metric"], _fmt(row["baseline"]), _fmt(row["verdict_current"])),
    )
    return Provocation("verdict", row["verdict_at"], prompt,
                       scorecard.ref(row["id"]), numbers, finding)


def _from_life(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """Somebody spoke on a server that had been silent for a month.

    Reads *yesterday's* ledger close, not today's. Recording the current reading
    and then reading it back in the same pass was a real bug in an earlier
    draft: the window stopped being all-zero at the exact moment the thing it
    existed to catch arrived. Yesterday's close cannot be erased by today's
    message.
    """
    if ledger.value_on(guild_id, ledger.day_offset(1), "activity.messages_30d") != 0.0:
        return None
    # NEWEST, not oldest. `since` is the fixed install mark, so ASC returned the
    # first message ever posted after installation — forever. Once that one was
    # journalled, `_already_thought` skipped it on every later pass and a server
    # that went quiet for a month and woke up again could never provoke a second
    # time. The ledger condition above is what bounds this: tomorrow's 30-day
    # close is no longer 0, so the wake-up stops being provokable within a day.
    row = db.one(
        "SELECT m.message_id, m.created_at, m.channel_id, "
        "       COALESCE(mem.display_name, mem.username) AS who "
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id=? AND m.created_at > ? AND m.type IN (0, 19) "
        "  AND COALESCE(mem.is_bot, 0) = 0 "
        "ORDER BY m.created_at DESC LIMIT 1",
        (guild_id, since),
    )
    if not row:
        return None
    prompt = "\n".join(
        [
            "# What happened",
            "",
            "Somebody posted. The 30-day message count closed at 0 yesterday, so this",
            "server had been completely silent for a month before this.",
            "",
            "  %s posted at %s" % (row["who"] or "a member", row["created_at"]),
            "",
            "# Your question",
            "",
            "Is this the start of something or a one-off? Say what you would need to",
            "see in the next week to tell those apart. Do not build a story on one",
            "message.",
        ]
    )
    return Provocation("life", row["created_at"], prompt, None, ())


def _from_join(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """Someone joined, and nobody had for a fortnight."""
    # A bot joining is an integration being added, not somebody arriving
    # somewhere quiet — and it would have produced "one person arrived, what
    # would keep them here in a month" about a webhook.
    row = db.one(
        "SELECT e.id, e.user_id, e.at, e.invite_code FROM member_events e "
        "LEFT JOIN members m ON m.guild_id = e.guild_id AND m.user_id = e.user_id "
        "WHERE e.guild_id=? AND e.event='join' AND e.at > ? "
        "  AND COALESCE(m.is_bot, 0) = 0 "
        "ORDER BY e.at DESC LIMIT 1",
        (guild_id, since),
    )
    if not row:
        return None
    # julianday on both sides, not `datetime(at, '-14 days')`. db.now() emits
    # '2026-08-10T19:31:00+00:00' while SQLite's datetime() returns
    # '2026-08-10 19:31:00' with a space, and ' ' sorts below 'T', so the naive
    # string comparison silently evaluates false. snapshot.py:135-140 documents
    # the same trap.
    # The drought window stops JOIN_BURST_DAYS before this join, not at it. A
    # shared link produces several arrivals in a few hours, and counting those
    # as "prior joins" made each one cancel the next: a burst — the single most
    # encouraging thing that can happen to a quiet server — provoked nothing at
    # all, while one lone join provoked normally.
    prior = db.scalar(
        "SELECT COUNT(*) FROM member_events WHERE guild_id=? AND event='join' "
        "AND julianday(at) < julianday(?) - %d "
        "AND julianday(at) >= julianday(?) - %d"
        % (JOIN_BURST_DAYS, JOIN_DROUGHT_DAYS),
        (guild_id, row["at"], row["at"]),
    )
    if prior:
        return None  # joins are routine here; not news
    # How many arrived together. The drought window deliberately ignores the
    # last JOIN_BURST_DAYS so a shared link is one arrival rather than several
    # cancelling each other — but the sentence then claimed "nobody had joined
    # in the 14 days before that" about the third of three people who joined
    # that afternoon, which is false and reads as false.
    burst = db.scalar(
        "SELECT COUNT(*) FROM member_events e "
        "LEFT JOIN members m ON m.guild_id = e.guild_id AND m.user_id = e.user_id "
        "WHERE e.guild_id=? AND e.event='join' AND COALESCE(m.is_bot,0)=0 "
        "AND julianday(e.at) >= julianday(?) - %d" % JOIN_BURST_DAYS,
        (guild_id, row["at"]),
    ) or 1
    arrival = ("Somebody joined at %s" % row["at"] if burst == 1
               else "%d people joined, the most recent at %s" % (burst, row["at"]))
    prompt = "\n".join(
        [
            "# What happened",
            "",
            "%s, after %d days in which nobody had."
            % (arrival, JOIN_DROUGHT_DAYS),
            "  invite attributed: %s" % (row["invite_code"] or "unknown"),
            "",
            "# Your question",
            "",
            "One person arrived somewhere quiet. What is the single thing that would",
            "most change whether they are still here in a month, and is it something",
            "the founder can do this week? If the honest answer is that one join says",
            "nothing, say that.",
        ]
    )
    return Provocation("joined", row["at"], prompt, None, ())


def _from_context(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """The founder edited the files that tell cadybot what the business is.

    Never surfaced. Reacting to somebody's own edit by messaging them about it
    is the behaviour of a tool nobody keeps installed. The value is that the
    next brief is written by a model that has already read the change.
    """
    newest = None
    for directory in (config.CONTEXT_DIR, config.PLAYBOOK_DIR):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.md")):
            stamp = db.iso(
                datetime.datetime.fromtimestamp(
                    path.stat().st_mtime, datetime.timezone.utc
                )
            )
            if newest is None or stamp > newest[0]:
                newest = (stamp, path)
    if newest is None or newest[0] <= since:
        return None
    stamp, path = newest
    prompt = "\n".join(
        [
            "# What happened",
            "",
            "`%s` was edited at %s. These files are the only thing you know about the"
            % (path.name, stamp),
            "business that is not visible in the server itself.",
            "",
            "# Your question",
            "",
            "Read it again as it stands now. Does anything you have been assuming",
            "about this business no longer hold? Nobody will see this answer — write",
            "the note you would want to have read before the next brief.",
        ]
    )
    return Provocation("context", stamp, prompt, None, ())


def _from_drift(guild_id: int, snap: Dict[str, Any], since: str) -> Optional[Provocation]:
    """A count moved materially over a fortnight, and a real event sits behind it.

    The second half is the important half. Requiring a message or a member event
    inside the window is what stops a number that decayed on its own — a 30-day
    count shedding old messages as the window slides — from reading as news.
    Decay is arithmetic, not information.

    `provoked_by` is the anchor event, never the metric, so several metrics
    moving off the same handful of messages collapse into one thought instead of
    buying one model call each.
    """
    moved = ledger.drift(guild_id)
    if moved is None:
        return None
    metric, before, after = moved

    # One thought per metric per fortnight, enforced here rather than by the
    # UNIQUE key. The key is (kind, provoked_by) and provoked_by is the newest
    # event in the window, so it advances every time anybody posts while the
    # drift itself — two ledger closes a fortnight apart — does not change at
    # all. Without this the same unchanged observation bought a fresh model call
    # on every tick for as long as the fortnight-old close differed, which on an
    # active server is permanently, and the desk ran at its daily ceiling
    # forever reporting the same thing.
    said_recently = db.one(
        "SELECT 1 FROM journal WHERE guild_id=? AND kind='drift' AND about_ref=? "
        "AND started_at >= ?",
        (guild_id, metric, db.days_ago(ledger.DRIFT_DAYS)),
    )
    if said_recently:
        return None

    window_start = db.days_ago(ledger.DRIFT_DAYS)
    anchor = db.one(
        "SELECT MAX(at) AS at FROM ("
        "  SELECT MAX(m.created_at) AS at FROM messages m LEFT JOIN members mem "
        "    ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "   WHERE m.guild_id=? AND m.created_at >= ? AND m.type IN (0, 19) "
        "     AND COALESCE(mem.is_bot, 0) = 0 "
        "  UNION ALL "
        "  SELECT MAX(at) FROM member_events WHERE guild_id=? AND at >= ?"
        ")",
        (guild_id, window_start, guild_id, window_start),
    )
    if not anchor or not anchor["at"] or anchor["at"] <= since:
        return None

    prompt = "\n".join(
        [
            "# What happened",
            "",
            "`%s` was %s a fortnight ago and is %s now."
            % (metric, _fmt(before), _fmt(after)),
            "The most recent thing that actually happened in this window was at %s."
            % anchor["at"],
            "",
            "This is a comparison of two stored daily closes. It is not a",
            "statistical test, and no significance is claimed for it.",
            "",
            "# Your question",
            "",
            "Is this a change in the server or a change in the arithmetic? Say which,",
            "and name the number you would look at next to tell. If a fortnight is",
            "too short to tell at this size, say so — that is the useful answer.",
        ]
    )
    # about_ref carries the metric so the per-fortnight check above has
    # something to match on.
    return Provocation("drift", anchor["at"], prompt, metric, _numerals(before, after))


_GENERATORS = {
    "ignored": _from_ignored,
    "backlog": _from_backlog,
    "verdict": _from_verdict,
    "life": _from_life,
    "joined": _from_join,
    "context": _from_context,
    "drift": _from_drift,
    "digest": _from_digest,
}


# --- selection -------------------------------------------------------------


def next_provocation(guild_id: int, snap: Dict[str, Any],
                     create_mark: bool = True) -> Optional[Provocation]:
    """The one thing worth thinking about, or None. Almost always None.

    Both time filters are ordinary `continue`s rather than assertions, and that
    is deliberate on two counts.

    A provocation dated in the future is not a bug, it is a race. Discord stamps
    message timestamps server-side, so a host clock running a few seconds slow
    makes the newest message look like it has not happened yet. As an assertion
    that crashed the pass, and it crashed it again on every tick for as long as
    that message stayed newest — a two-second clock skew silently disabled the
    whole desk. Skipping instead costs one cycle: the row is still there in six
    hours, and by then it is unambiguously in the past.

    Assertions are also the wrong tool for a load-bearing rule, because `python
    -O` deletes them. The check that a generator never reaches for `db.now()`
    belongs where it can be exhaustive and cannot be optimised away: case 10 in
    tests/thinker.py runs it against every generator on controlled fixtures.
    """
    started_at = db.now()
    installed = installed_at(guild_id, create=create_mark)
    if installed is None:
        return None      # nothing is installed and we were told not to install

    for kind in KINDS:
        prov = _GENERATORS[kind](guild_id, snap, installed)
        if prov is None:
            continue
        if prov.provoked_by >= started_at:
            continue  # hasn't happened yet by this clock; look again next tick
        if prov.provoked_by <= installed and kind not in LOOKS_BACK:
            continue  # older than the agenda: history, not news
        if _already_thought(guild_id, prov):
            continue
        return prov
    return None


def _already_thought(guild_id: int, prov: Provocation) -> bool:
    """Has this exact provocation been dealt with?

    A failure that still has attempts left does not count as dealt with, or the
    retry `open_attempt` is prepared to do could never be reached — this runs
    first.
    """
    return bool(
        db.one(
            "SELECT 1 FROM journal WHERE guild_id=? AND kind=? AND provoked_by=? "
            "AND NOT (outcome='failed' AND attempts < ?)",
            (guild_id, prov.kind, prov.provoked_by, MAX_ATTEMPTS),
        )
    )


# --- budget ----------------------------------------------------------------


def affordable(guild_id: int) -> bool:
    """Is there budget for one more thought today?

    Counts `journal` rows, not `runs` rows. `runs` is written only on the
    success paths — `llm._ollama` raises BackendError on a read timeout before
    `record_local_run`, and `llm._claude` raises on `max_tokens` after a fully
    billed response — so a budget read from `runs` never charges for the calls
    that cost the most and fail. The journal row is written before the call, so
    failures are charged and a crash loop cannot spin.
    """
    if config.THINK_CALLS_PER_DAY <= 0:
        return False
    spent = db.scalar(
        "SELECT COUNT(*) FROM journal WHERE guild_id=? AND started_at>=?",
        (guild_id, db.hours_ago(24)),
    )
    return spent < config.THINK_CALLS_PER_DAY


# --- the journal -----------------------------------------------------------


def open_attempt(guild_id: int, prov: Provocation) -> Optional[int]:
    """Claim this provocation, before spending anything on it.

    Returns the row id, or None if somebody else already has it. The UNIQUE
    constraint is doing the locking; because `provoked_by` came out of storage,
    a racing process built the identical key and one of the two INSERTs loses.
    """
    try:
        cur = db.connect().execute(
            "INSERT INTO journal (guild_id, kind, provoked_by, about_ref, "
            "                     started_at, self_prompt, finding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                guild_id,
                prov.kind,
                prov.provoked_by,
                prov.about_ref,
                db.now(),
                prov.self_prompt,
                # Persisted because the finding is written at provocation time
                # and delivered on a later tick. Recomputing it at delivery
                # would re-run SQL against a database that has moved on; storing
                # it means the founder is told what was true when it was noticed.
                prov.finding,
            ),
        )
    except sqlite3.IntegrityError:
        # The row exists. That is normally the point — a provocation is consumed
        # once. But if the previous attempt died because the backend was down or
        # timed out, consuming it means an ollama restart or a cold model on the
        # one tick that mattered silently destroys the thought forever. Give a
        # failure exactly MAX_ATTEMPTS goes, then let it rest: a provocation that
        # cannot be thought about twice cannot be thought about a hundred times
        # either.
        retry = db.one(
            "SELECT id, attempts FROM journal WHERE guild_id=? AND kind=? "
            "AND provoked_by=? AND outcome='failed' AND attempts < ?",
            (guild_id, prov.kind, prov.provoked_by, MAX_ATTEMPTS),
        )
        if retry is None:
            return None
        db.connect().execute(
            "UPDATE journal SET outcome='started', failure=NULL, attempts=attempts+1, "
            "       started_at=? WHERE id=?",
            (db.now(), retry["id"]),
        )
        return retry["id"]
    return cur.lastrowid


def record_result(
    journal_id: int, result: Any, unverified: List[str], model: Optional[str]
) -> None:
    db.connect().execute(
        "UPDATE journal SET outcome='thought', restated=?, reasoning=?, evidence=?, "
        "       note_to_self=?, watch_metric=?, to_founder=?, unverified=?, "
        "       wanted_telling=?, model=? WHERE id=?",
        (
            getattr(result, "restated", None),
            getattr(result, "reasoning", None),
            getattr(result, "evidence", None),
            getattr(result, "note_to_self", None),
            getattr(result, "watch_metric", None),
            getattr(result, "to_founder", None),
            json.dumps(unverified or []),
            1 if getattr(result, "worth_telling_founder", False) else 0,
            model,
            journal_id,
        ),
    )


def record_failure(journal_id: int, exc: BaseException) -> None:
    """A failed thought stays on the record, and stays charged."""
    db.connect().execute(
        "UPDATE journal SET outcome='failed', failure=? WHERE id=?",
        (("%s: %s" % (type(exc).__name__, exc))[:500], journal_id),
    )


def mark_surfaced(journal_id: int) -> None:
    db.connect().execute(
        "UPDATE journal SET surfaced_at=? WHERE id=?", (db.now(), journal_id)
    )


# --- speaking --------------------------------------------------------------


def may_surface(guild_id: int, prov: Provocation) -> Tuple[bool, str]:
    """May a thought of this kind reach the founder right now?

    Returns (allowed, reason) so `cadybot reflect` and `/notes` can show why not.
    Every clause is deterministic and none of them consults a model — the model
    gets a veto over this decision and never a vote for it. Asked "is this worth
    saying", a model says yes nearly always; the only direction in which that
    bias is harmless is refusal.
    """
    if prov.kind not in SURFACEABLE:
        return False, "%s thoughts are never surfaced" % prov.kind

    if config.BACKEND not in config.THINK_SURFACE_BACKENDS:
        return False, (
            "backend %r may think but not speak — unprompted speech needs the "
            "model the stage gates were written for" % config.BACKEND
        )

    quiet = db.get_setting(guild_id, QUIET_KEY)
    if quiet and quiet > db.now():
        return False, "quiet until %s" % quiet

    hour = int(db.now()[11:13])
    low, high = config.SURFACE_WINDOW_UTC
    if not (low <= hour < high):
        return False, "outside the %02d:00-%02d:00 UTC window" % (low, high)

    if db.deliveries_since(guild_id, db.hours_ago(config.SURFACE_MIN_GAP_HOURS)):
        return False, "cadybot spoke within the last %dh" % config.SURFACE_MIN_GAP_HOURS

    said = db.deliveries_since(guild_id, db.days_ago(7), "thought")
    if said >= config.SURFACE_MAX_PER_WEEK:
        return False, "already volunteered %d thought(s) this week" % said

    return True, "allowed"


# --- reading back ----------------------------------------------------------


def reap_stale(guild_id: int, older_than_hours: int = 2) -> int:
    """Close out thoughts whose process died mid-call.

    `open_attempt` writes the row before the model call so a crash still costs
    its budget, and `record_failure` closes it when Python sees the exception.
    A SIGTERM from `systemctl restart`, an OOM kill, or a laptop lid does not
    raise anything — the row stays at 'started' forever, and because the
    provocation is keyed by (kind, provoked_by) it can never be raised again.
    A routine restart during the one model call of the day therefore destroyed
    that thought permanently. Two hours is well past the 600s backend timeout.
    """
    return db.connect().execute(
        "UPDATE journal SET outcome='failed', "
        "       failure='abandoned — the process died during the call' "
        "WHERE guild_id=? AND outcome='started' AND started_at < ?",
        (guild_id, db.hours_ago(older_than_hours)),
    ).rowcount


def unsaid(guild_id: int, within_hours: int = 48) -> Optional[Dict[str, Any]]:
    """A thought that wanted to be told and has not been, yet.

    The speaking gates are mostly temporal — an afternoon window, a 20-hour gap,
    one a week — and they were being evaluated *after* the provocation was
    claimed and the model call paid for. Three ticks in four fall outside the
    window, so the usual outcome was: think about the closed bet, decide it is
    worth telling, fail the clock, and lose it forever, because the provocation
    was already consumed and could never be raised again.

    Keeping it eligible costs nothing and needs no second model call — the
    sentence is already written and already checked. It simply waits for a tick
    that is allowed to speak. After `within_hours` it is stale news and stays
    unsaid, which is the right outcome for something two days old.
    """
    return db.one(
        "SELECT * FROM journal WHERE guild_id=? AND outcome='thought' "
        "AND surfaced_at IS NULL AND to_founder IS NOT NULL AND to_founder <> '' "
        "AND wanted_telling=1 AND (unverified IS NULL OR unverified IN ('', '[]')) "
        # Only kinds that could ever be said. A `context` thought is never
        # surfaceable, so holding one here head-of-line blocks the single replay
        # slot forever and a real finding queues behind it permanently.
        "AND kind IN (%s) " % ",".join("'%s'" % k for k in SURFACEABLE) +
        # A thought carrying a SQL-written finding outranks one that is only
        # prose, however recent. Otherwise a generic reflection generated
        # minutes later buries the one that actually found something.
        "AND started_at >= ? ORDER BY (finding IS NOT NULL) DESC, id DESC LIMIT 1",
        (guild_id, db.hours_ago(within_hours)),
    )


def live_notes(guild_id: int, limit: Optional[int] = None) -> List[str]:
    """The notes cadybot left itself, for the next brief to read.

    Capped and expiring, so carried context cannot grow with uptime.
    """
    rows = db.query(
        "SELECT note_to_self FROM journal WHERE guild_id=? AND outcome='thought' "
        "AND note_to_self IS NOT NULL AND note_to_self <> '' AND started_at >= ? "
        "ORDER BY id DESC LIMIT ?",
        (guild_id, db.days_ago(config.NOTE_TTL_DAYS), limit or config.NOTES_CARRIED),
    )
    return [r["note_to_self"] for r in rows]


def known_numbers(prov: Provocation) -> set:
    """Figures this module injected into the prompt.

    Subtracted from verify_evidence's output so the quality gate does not flag
    cadybot for quoting a number cadybot handed it.

    The reference itself has to be in here. `advisor._NUMERAL` is
    `-?\\d+(?:\\.\\d+)?`, so the perfectly ordinary sentence "R-1 closed without
    being tried" yields the token `-1`, which appears in no snapshot — and
    thinking.py refuses to volunteer a thought carrying an unverified figure.
    The effect was that the single most valuable kind of thought, the one about
    a closed bet, could never be spoken at all. Silently: it was journalled
    every time and surfaced never.
    """
    known = set(prov.numbers)
    if prov.about_ref:
        for token in _NUMERAL.findall(prov.about_ref):
            known.add(token)
            known.add(token.lstrip("-"))
    return known


def recent(guild_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Everything thought lately, surfaced or not. Backs /notes."""
    return db.query(
        "SELECT id, kind, provoked_by, about_ref, started_at, outcome, failure, "
        "       note_to_self, to_founder, surfaced_at, watch_metric, unverified, "
        "       wanted_telling "
        "FROM journal WHERE guild_id=? ORDER BY id DESC LIMIT ?",
        (guild_id, limit),
    )


# --- helpers ---------------------------------------------------------------


def _flat_numbers(node: Any) -> List[Any]:
    """Every numeric leaf in a findings dict, so the model may cite them."""
    out: List[Any] = []
    if isinstance(node, dict):
        for v in node.values():
            out.extend(_flat_numbers(v))
    elif isinstance(node, (list, tuple)):
        for v in node:
            out.extend(_flat_numbers(v))
    elif isinstance(node, bool):
        return out
    elif isinstance(node, (int, float)):
        out.append(node)
    return out


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return "%d" % value
    return "%.3g" % value


def _numerals(*values) -> Tuple[str, ...]:
    """Normalised string forms of code-supplied figures, matching advisor._normalise."""
    out = []
    for value in values:
        if value is None:
            continue
        for text in (_fmt(value), str(value), repr(float(value))):
            token = text.lstrip("+")
            if "." in token:
                token = token.rstrip("0").rstrip(".")
            if token and token not in out:
                out.append(token)
    return tuple(out)
