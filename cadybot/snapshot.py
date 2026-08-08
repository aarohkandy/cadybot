"""Deterministic server state. No LLM touches this file.

Claude never computes a number — it only explains and prescribes on top of what
this module produces. That is what stops it inventing statistics.

Every rate here is gated on its own denominator rather than on the server's
stage, so metrics switch themselves on as the server grows instead of waiting
for someone to remember them. Below stats.MIN_RATE_DENOMINATOR a rate is a pair
of counts and carries no percentage at all.

Two things Discord does not expose, recorded here so nobody researches them a
third time. Visitors (members who opened a channel without posting) and
Discord's own retention curves are both computed from client-side channel-view
telemetry, which no bot API returns at any permission level. Server Insights is
dashboard-only: the VIEW_GUILD_INSIGHTS permission bit (1<<19) exists, but no
REST route consumes it, and the proposed GET /guilds/{id}/analytics was never
implemented. They are reported as explicit nulls in `not_measurable` so the
model has a fact to refuse with instead of a plausible-looking invention.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from . import config, db, room, stats

# The floors live in config for the rest of the codebase and in stats for the
# functions that enforce them. Neither imports the other, so the agreement is
# checked here, once, where both are already in scope.
assert stats.MIN_RATE_DENOMINATOR == config.MIN_RATE_DENOMINATOR
assert stats.MIN_RECIPROCITY_POSTERS == config.MIN_RECIPROCITY_POSTERS
assert stats.MIN_RECIPROCITY_EDGES == config.MIN_RECIPROCITY_EDGES
assert stats.MIN_VERDICT_EVENTS == config.MIN_VERDICT_EVENTS


def _stage(humans: int) -> str:
    if humans < 25:
        return "seed"
    if humans < 100:
        return "sprout"
    if humans < 500:
        return "growing"
    return "community"


def _parse(iso_ts: Optional[str]) -> Optional[datetime]:
    if not iso_ts:
        return None
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return then


def _age_days(iso_ts: Optional[str]) -> Optional[float]:
    then = _parse(iso_ts)
    if then is None:
        return None
    return round((datetime.now(timezone.utc) - then).total_seconds() / 86400, 1)


def _name(row: Any) -> str:
    return row["display_name"] or row["username"] or str(row["user_id"])


# Discord's MessageType values that represent a human saying something: 0 is an
# ordinary message and 19 a reply. Everything else in the enum is Discord
# narrating its own events — 21 in particular is the content-free mirror posted
# in the parent channel when a thread is created, authored by whoever started
# the thread. Counting those inflates every number in this file, and the error
# grows in exact proportion to how much the server uses threads. Types 20 and 23
# are application-command responses and are bot-authored, so the is_bot filter
# already removes them; they are excluded here anyway so the gate reads the same
# everywhere.
HUMAN_TYPES = "(0, 19)"

# Channel kinds a founder can restructure. Threads and categories are excluded
# deliberately -- see the comment at the channels query.
PRUNABLE_KINDS = (
    "('TextChannel', 'VoiceChannel', 'ForumChannel', 'StageChannel', 'NewsChannel')"
)

# A snapshot must stay roughly the same size whether the server has 7 members
# or 50,000, because it has to fit in a context window either way. Anything
# that grows with member count is reported as a count plus a sample; the model
# is told the difference so it never mistakes a sample for the whole list.
SAMPLE_MEMBERS = 12
SAMPLE_CHANNELS = 25
SAMPLE_PAIRS = 20
SAMPLE_COHORTS = 8

# The power-user histogram is a fixed 29 slots (0 through 28 active days) at
# every server size, which is what lets it replace DAU/MAU. DAU/MAU is quantized
# to 1/MAU and moves 14.3 percentage points per person at MAU=7.
ACTIVE_DAYS_WINDOW = 28

# Top-level blocks that vanish rather than render when their own denominator is
# too thin. Everything else keeps its key even when the value is null.
GATED_BLOCKS = ("top_share_30d",)


def _capped(items: List[Any], limit: int, total: Optional[int] = None) -> Dict[str, Any]:
    """Count plus a bounded sample. `total` is for lists already LIMITed in SQL."""
    n = len(items) if total is None else total
    out: Dict[str, Any] = {"count": n}
    if n <= limit and len(items) >= n:
        out["all"] = items
    else:
        shown = items[:limit]
        out["sample"] = shown
        out["note"] = (
            "this is a sample of %d; the real total is %d — do not count the sample"
            % (len(shown), n)
        )
    return out


def _unreadable(reason: str) -> Dict[str, Any]:
    """A fact cadybot could not read. Never collapse this to 0 or False."""
    return {"value": None, "reason": reason}


NO_PERMISSION = "cadybot lacks the permission to read this"
NOT_SAMPLED = "cadybot has not read this server's configuration yet"


# How long after a question a later message still counts as its answer, as a
# fraction of a day, ready to be added to a julianday().
_ANSWER_WINDOW_DAYS = "%.8f" % (config.RESPONSE_WINDOW_HOURS / 24.0)

# Whether anyone other than the asker answered, within the response window.
#
# Two traps are dodged here. The window is expressed with julianday on both
# sides rather than as a string comparison against datetime(): db.now() emits
# '2026-08-04T17:31:00.123456+00:00' while SQLite's datetime() returns
# '2026-08-04 19:31:00' — space separator, no offset — and ' ' sorts below 'T',
# so `created_at < datetime(created_at, '+2 hours')` silently evaluates FALSE.
# And the reply may not be in the same channel: a thread created from a message
# has thread.id == starter_message.id, so a reply to the question can live in a
# channel whose id is the question's own message_id, or in any thread parented
# to the question's channel. The type gate above must stay applied to `r`, or
# the thread's own type-21 starter mirror — authored by the original asker but
# recorded against the thread — would masquerade as somebody answering.
_ANSWERED = (
    "EXISTS (SELECT 1 FROM messages r "
    "        WHERE r.guild_id = m.guild_id "
    "          AND r.type IN " + HUMAN_TYPES + " "
    "          AND r.author_id != m.author_id "
    "          AND julianday(r.created_at) > julianday(m.created_at) "
    "          AND julianday(r.created_at) <= julianday(m.created_at) + " + _ANSWER_WINDOW_DAYS + " "
    "          AND (r.channel_id = m.channel_id "
    "               OR r.channel_id = m.message_id "
    "               OR r.channel_id IN (SELECT channel_id FROM channels "
    "                                   WHERE guild_id = m.guild_id AND kind = 'Thread' "
    "                                     AND parent_id = m.channel_id)))"
)

# Non-owner, non-bot, human-typed messages containing a question mark. Crude,
# and correct often enough to be the single most actionable alert at this scale.
# Split into a FROM and a WHERE so a caller can slide another join between them.
_QUESTION_FROM = (
    "FROM messages m "
    "LEFT JOIN members mem ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
)
_QUESTION_WHERE = (
    "WHERE m.guild_id = ? AND m.author_id != ? "
    "  AND COALESCE(mem.is_bot, 0) = 0 "
    "  AND m.type IN " + HUMAN_TYPES + " "
    "  AND m.content LIKE '%?%' "
    "  AND m.created_at >= ? "
)


def unanswered_questions(guild_id: int, owner_id: int) -> List[Dict[str, Any]]:
    """Member questions nobody replied to, most recent first."""
    rows = db.query(
        "SELECT m.message_id, m.channel_id, m.author_id, m.created_at, m.content, "
        "       c.name AS channel_name, "
        "       COALESCE(mem.display_name, mem.username) AS author_name, "
        # Two independent tells that the question spawned a thread: Discord's
        # HAS_THREAD message flag, and a channels row whose id is the question's
        # own message_id. The flag is unset on everything ingested before the
        # thread work landed, so the channels branch is the one that carries it.
        "       ((m.flags & 32) != 0 OR EXISTS (SELECT 1 FROM channels t "
        "            WHERE t.guild_id = m.guild_id AND t.kind = 'Thread' "
        "              AND t.channel_id = m.message_id)) AS spawned_thread "
        + _QUESTION_FROM
        + "LEFT JOIN channels c ON c.guild_id = m.guild_id AND c.channel_id = m.channel_id "
        + _QUESTION_WHERE
        + "  AND NOT " + _ANSWERED + " "
        "ORDER BY m.created_at DESC LIMIT 10",
        (guild_id, owner_id, db.days_ago(30)),
    )
    out = []
    for r in rows:
        age = _age_days(r["created_at"])
        if age is None or age * 24 < config.UNANSWERED_HOURS:
            continue  # still fresh; not yet a failure
        out.append(
            {
                "author": r["author_name"] or str(r["author_id"]),
                "channel": r["channel_name"],
                "asked_days_ago": age,
                "spawned_thread": bool(r["spawned_thread"]),
                "text": (r["content"] or "")[:400],
                "link": "https://discord.com/channels/%d/%d/%d"
                % (guild_id, r["channel_id"], r["message_id"]),
            }
        )
    return out


def _response_rate(guild_id: int, owner_id: int) -> Dict[str, Any]:
    """Answered-question rate over a censoring-complete denominator.

    Only questions older than the response window are counted, because only
    those have a known outcome. No mean or median latency is reported anywhere:
    a latency averaged over the answered subset is survivorship bias on
    right-censored data, and it is systematically optimistic on exactly the
    servers where slow responses are the problem.
    """
    row = db.one(
        "SELECT COUNT(*) AS asked, "
        "       SUM(CASE WHEN " + _ANSWERED + " THEN 1 ELSE 0 END) AS answered "
        + _QUESTION_FROM
        + _QUESTION_WHERE
        + "  AND julianday(m.created_at) <= julianday('now') - " + _ANSWER_WINDOW_DAYS,
        (guild_id, owner_id, db.days_ago(30)),
    )
    asked = (row["asked"] if row else 0) or 0
    answered = (row["answered"] if row else 0) or 0
    rendered = stats.render_rate(answered, asked)
    out: Dict[str, Any] = {"answered": answered, "asked": asked}
    for key in ("pct", "ci_low", "ci_high"):
        if key in rendered:
            out[key] = rendered[key]
    out["window_hours"] = config.RESPONSE_WINDOW_HOURS
    out["note"] = (
        "Denominator holds only questions older than the response window, so "
        "every one has a settled outcome. No latency is reported: averaging "
        "over answered questions alone is survivorship bias."
    )
    return out


def _burst_counts(guild_id: int, since: str) -> Dict[int, int]:
    """Messages per author, collapsing each run of consecutive same-author
    messages inside BURST_MINUTES into one.

    Raw message counts let a single two-person back-and-forth read as one member
    dominating the server. A burst is one turn at speaking, which is the unit a
    share-of-conversation is actually about.
    """
    rows = db.query(
        "SELECT m.channel_id, m.author_id, m.created_at "
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id = ? AND COALESCE(mem.is_bot, 0) = 0 "
        "  AND m.type IN " + HUMAN_TYPES + " AND m.created_at >= ? "
        "ORDER BY m.channel_id, m.created_at",
        (guild_id, since),
    )
    gap = timedelta(minutes=config.BURST_MINUTES)
    counts: Dict[int, int] = {}
    last_channel: Optional[int] = None
    last_author: Optional[int] = None
    last_at: Optional[datetime] = None
    for r in rows:
        at = _parse(r["created_at"])
        continues = (
            r["channel_id"] == last_channel
            and r["author_id"] == last_author
            and at is not None
            and last_at is not None
            and at - last_at <= gap
        )
        if not continues:
            counts[r["author_id"]] = counts.get(r["author_id"], 0) + 1
        last_channel, last_author, last_at = r["channel_id"], r["author_id"], at
    return counts


def _share(numerator: int, denominator: int) -> Dict[str, Any]:
    low, high = stats.wilson(numerator, denominator)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "share": round(numerator / float(denominator), 3),
        "ci_low": round(low, 3),
        "ci_high": round(high, 3),
    }


def _top_share(guild_id: int, owner_id: int) -> Optional[Dict[str, Any]]:
    """Concentration of conversation, denominated in bursts rather than humans.

    This is the one inequality metric that already works at seven members,
    because its denominator is turns at speaking — hundreds of them — not the
    seven people. Gini, a 90-9-1 split and a power-law exponent all need a
    member-count denominator and are not computed anywhere in this file.
    """
    counts = _burst_counts(guild_id, db.days_ago(30))
    total = sum(counts.values())
    if total < config.MIN_SHARE_DENOMINATOR:
        return None
    ordered = sorted(counts.values(), reverse=True)
    return {
        "unit": "bursts (consecutive messages from one author within %d minutes count once)"
        % config.BURST_MINUTES,
        "top1": _share(ordered[0], total),
        "top3": _share(sum(ordered[:3]), total),
        "owner": _share(counts.get(owner_id, 0), total),
    }


def _active_days(guild_id: int, humans: int) -> Dict[str, Any]:
    """How many members were active on how many distinct days in 28.

    Replaces DAU/MAU, which at MAU=7 is quantized to 1/7 and jumps 14.3
    percentage points when one person opens the app. This is a fixed-width
    vector of 29 integers at any server size, and every entry is a plain count.
    """
    histogram = [0] * (ACTIVE_DAYS_WINDOW + 1)
    posted = 0
    for r in db.query(
        "SELECT active_days, COUNT(*) AS members FROM ("
        "  SELECT m.author_id, COUNT(DISTINCT substr(m.created_at, 1, 10)) AS active_days "
        "  FROM messages m LEFT JOIN members mem "
        "    ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "  WHERE m.guild_id = ? AND m.created_at >= ? "
        "    AND m.type IN " + HUMAN_TYPES + " AND COALESCE(mem.is_bot, 0) = 0 "
        "  GROUP BY m.author_id"
        ") GROUP BY active_days",
        (guild_id, db.days_ago(ACTIVE_DAYS_WINDOW)),
    ):
        slot = min(int(r["active_days"]), ACTIVE_DAYS_WINDOW)
        histogram[slot] += r["members"]
        posted += r["members"]
    histogram[0] = max(0, humans - posted)
    return {
        "window_days": ACTIVE_DAYS_WINDOW,
        "members_by_active_days": histogram,
        "note": "Index i is how many human members posted on exactly i distinct "
        "days in the window. Index 0 includes members who never posted at all.",
    }


def _threads(guild_id: int) -> Dict[str, Any]:
    """Thread lifecycle. Threads are identified by kind, never by parent_id.

    parent_id is overloaded: on a Thread it is the parent channel, on an
    ordinary TextChannel it is the enclosing category. Counting rows with a
    parent_id would pull real channels into the thread accounting.
    """
    archived_counts = [
        r["thread_message_count"]
        for r in db.query(
            "SELECT thread_message_count FROM channels "
            "WHERE guild_id = ? AND kind = 'Thread' AND COALESCE(archived, 0) = 1 "
            "  AND thread_message_count IS NOT NULL",
            (guild_id,),
        )
    ]
    return {
        "opened_30d": db.scalar(
            "SELECT COUNT(*) FROM channels WHERE guild_id = ? AND kind = 'Thread' "
            "AND created_at >= ?",
            (guild_id, db.days_ago(30)),
        ),
        "total": db.scalar(
            "SELECT COUNT(*) FROM channels WHERE guild_id = ? AND kind = 'Thread'",
            (guild_id,),
        ),
        "still_active": db.scalar(
            "SELECT COUNT(*) FROM channels WHERE guild_id = ? AND kind = 'Thread' "
            "AND COALESCE(archived, 0) = 0",
            (guild_id,),
        ),
        "archived": db.scalar(
            "SELECT COUNT(*) FROM channels WHERE guild_id = ? AND kind = 'Thread' "
            "AND COALESCE(archived, 0) = 1",
            (guild_id,),
        ),
        "median_messages_in_archived": stats.median(archived_counts),
        # Splitting these needs the id of whoever archived the thread. Discord
        # sends it, but nothing stores it, so the split is unreadable rather
        # than zero — "no thread was ever archived deliberately" is a very
        # different claim from "cadybot cannot tell which were".
        "archived_by_timeout": _unreadable(
            "channels has no archiver_id column, so a thread that timed out "
            "cannot be told from one somebody closed"
        ),
        "archived_deliberately": _unreadable(
            "channels has no archiver_id column, so a thread that timed out "
            "cannot be told from one somebody closed"
        ),
    }


def _structure(guild_id: int, owner_id: int, posters_30: int) -> Dict[str, Any]:
    """Who talks to whom. Every list here is capped; none may grow with members."""
    since = db.days_ago(30)

    only_owner_total = db.scalar(
        "SELECT COUNT(*) FROM ("
        "  SELECT mentioned_id FROM mentions WHERE guild_id = ? AND created_at >= ? "
        "    AND mentioned_id != ? AND mentioned_id != author_id "
        "  GROUP BY mentioned_id "
        "  HAVING SUM(CASE WHEN author_id = ? THEN 1 ELSE 0 END) = COUNT(*)"
        ")",
        (guild_id, since, owner_id, owner_id),
    )
    only_owner = [
        r["who"] or str(r["mentioned_id"])
        for r in db.query(
            "SELECT n.mentioned_id, COALESCE(mem.display_name, mem.username) AS who "
            "FROM (SELECT mentioned_id FROM mentions "
            "      WHERE guild_id = ? AND created_at >= ? "
            "        AND mentioned_id != ? AND mentioned_id != author_id "
            "      GROUP BY mentioned_id "
            "      HAVING SUM(CASE WHEN author_id = ? THEN 1 ELSE 0 END) = COUNT(*)) n "
            "LEFT JOIN members mem ON mem.guild_id = ? AND mem.user_id = n.mentioned_id "
            "LIMIT ?",
            (guild_id, since, owner_id, owner_id, guild_id, SAMPLE_MEMBERS),
        )
    ]

    # The reply graph. Bounded by replies in the window, not by members squared,
    # because it is built from actual reply edges rather than from every pair.
    dyad_rows = db.query(
        "SELECT r.author_id AS src, p.author_id AS dst, COUNT(*) AS n, "
        "       COALESCE(ma.display_name, ma.username) AS src_name, "
        "       COALESCE(mb.display_name, mb.username) AS dst_name "
        "FROM messages r "
        "JOIN messages p ON p.guild_id = r.guild_id AND p.message_id = r.reply_to_id "
        "LEFT JOIN members ma ON ma.guild_id = r.guild_id AND ma.user_id = r.author_id "
        "LEFT JOIN members mb ON mb.guild_id = r.guild_id AND mb.user_id = p.author_id "
        "WHERE r.guild_id = ? AND r.created_at >= ? "
        "  AND r.type IN " + HUMAN_TYPES + " AND p.type IN " + HUMAN_TYPES + " "
        "  AND COALESCE(ma.is_bot, 0) = 0 AND COALESCE(mb.is_bot, 0) = 0 "
        "  AND r.author_id != p.author_id "
        "GROUP BY src, dst ORDER BY n DESC",
        (guild_id, since),
    )
    edges: Dict[Tuple[int, int], int] = {
        (r["src"], r["dst"]): r["n"] for r in dyad_rows
    }
    dyads = [
        {
            "from": r["src_name"] or str(r["src"]),
            "to": r["dst_name"] or str(r["dst"]),
            "replies_30d": r["n"],
        }
        for r in dyad_rows[:SAMPLE_PAIRS]
    ]

    return {
        "mentions_30d": db.scalar(
            "SELECT COUNT(*) FROM mentions WHERE guild_id = ? AND created_at >= ?",
            (guild_id, since),
        ),
        "mention_everyone_30d": db.scalar(
            "SELECT COUNT(*) FROM messages m LEFT JOIN members mem "
            "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
            "WHERE m.guild_id = ? AND m.created_at >= ? AND m.mention_everyone = 1 "
            "  AND m.type IN " + HUMAN_TYPES + " AND COALESCE(mem.is_bot, 0) = 0",
            (guild_id, since),
        ),
        # Sharper than never_posted: it separates a member nobody thinks about
        # from one who is simply quiet but gets talked to.
        "mentioned_only_by_owner": _capped(only_owner, SAMPLE_MEMBERS, total=only_owner_total),
        "reply_dyads_30d": _capped(dyads, SAMPLE_PAIRS, total=len(dyad_rows)),
        "reciprocity": stats.reciprocity(edges, posters_30),
        "reciprocity_note": "Density-corrected (Garlaschelli & Loffredo). null "
        "means the reply graph is too small to carry the statistic, not that "
        "reciprocity is zero.",
    }


def _server_config(guild_id: int, humans_present: int) -> Dict[str, Any]:
    """Guild-level settings, with unreadable kept distinct from off."""
    facts = db.one("SELECT * FROM guild_facts WHERE guild_id = ?", (guild_id,))
    keys = facts.keys() if facts is not None else []

    def fact(column: str) -> Any:
        if facts is None:
            return _unreadable(NOT_SAMPLED)
        if column not in keys or facts[column] is None:
            return _unreadable(NO_PERMISSION)
        return facts[column]

    onboarding_enabled = fact("onboarding_enabled")
    out: Dict[str, Any] = {
        "onboarding_enabled": onboarding_enabled,
        "onboarding_mode": fact("onboarding_mode"),
        "onboarding_prompts": fact("onboarding_prompts"),
        "onboarding_required_prompts": fact("onboarding_required_prompts"),
        "onboarding_default_channels": fact("onboarding_default_channels"),
        "widget_enabled": fact("widget_enabled"),
        "boost_count": fact("boost_count"),
        "boost_tier": fact("boost_tier"),
        "verification_level": fact("verification_level"),
        "is_community": fact("is_community"),
        # The two flags that say which of the nulls above are permission
        # failures rather than settings cadybot has not looked at yet. Written
        # by the listener since ingest landed and never surfaced until now, so
        # every unreadable fact came out with the same generic reason.
        "onboarding_readable": fact("onboarding_readable"),
        "audit_readable": fact("audit_readable"),
    }

    # Onboarding completion is absent, not 0%, when onboarding is off: a server
    # with the feature disabled would otherwise read exactly like one where
    # every arrival abandoned the funnel, and those call for opposite advice.
    if onboarding_enabled == 1:
        started = db.scalar(
            "SELECT COUNT(*) FROM members WHERE guild_id = ? AND is_bot = 0 "
            "AND left_at IS NULL",
            (guild_id,),
        )
        completed = db.scalar(
            "SELECT COUNT(*) FROM members WHERE guild_id = ? AND is_bot = 0 "
            "AND left_at IS NULL AND pending = 0",
            (guild_id,),
        )
        out["onboarding_completion"] = stats.render_rate(completed, started)
        out["onboarding_completion_note"] = (
            "Derived from members.pending, which Discord flips exactly once when "
            "someone finishes onboarding. Discord exposes no funnel counters."
        )

    # Lifetime, never windowed: someone who left and came back is a fact about
    # the server, not about the last 30 days.
    out["boomerang_members"] = db.scalar(
        "SELECT COUNT(DISTINCT j.user_id) FROM member_events j "
        "WHERE j.guild_id = ? AND j.event = 'join' AND EXISTS ("
        "  SELECT 1 FROM member_events l WHERE l.guild_id = j.guild_id "
        "    AND l.user_id = j.user_id AND l.event = 'leave' AND l.at < j.at)",
        (guild_id,),
    )
    out["boomerang_note"] = (
        "Derived from join events that follow a leave event, so it only sees "
        "rejoins that happened after logging_since."
    )
    out["pending_members"] = db.scalar(
        "SELECT COUNT(*) FROM members WHERE guild_id = ? AND is_bot = 0 "
        "AND left_at IS NULL AND pending = 1",
        (guild_id,),
    )
    out["members_counted_for_stage"] = humans_present

    online = [
        r["approx_online"]
        for r in db.query(
            "SELECT approx_online FROM presence_samples "
            "WHERE guild_id = ? AND at >= ? AND approx_online IS NOT NULL",
            (guild_id, db.days_ago(7)),
        )
    ]
    if online:
        out["online_7d"] = {
            "min": min(online),
            "median": stats.median(online),
            "max": max(online),
            "samples": len(online),
        }
    else:
        out["online_7d"] = _unreadable(
            "no presence samples in the last 7 days" if facts is not None else NOT_SAMPLED
        )
    return out


# Discord's AuditLogAction values for the moderation slice listener.py stores.
# The numbers are restated rather than imported: this file must stay runnable
# without discord.py so the harness can build a snapshot from a bare database.
AUDIT_ACTIONS = (("kicks", 20), ("prunes", 21), ("bans", 22), ("unbans", 23),
                 ("automod_blocks", 143))


def _moderation(guild_id: int) -> Dict[str, Any]:
    """Departures somebody caused, separated from departures that just happened.

    membership_flow_30d counts leaves and cannot tell why. Sixty people leaving
    a server is a crisis; sixty people being banned in the same window is a raid
    that was dealt with, and the advice for the two is opposite. This is the only
    thing that distinguishes them, and cadybot has been recording it since ingest
    landed without anything reading it back.
    """
    since = db.days_ago(30)
    readable = db.scalar(
        "SELECT audit_readable FROM guild_facts WHERE guild_id = ?",
        (guild_id,),
        default=None,
    )
    if readable == 0:
        return {"available": False, "reason": NO_PERMISSION}
    counts = {
        name: db.scalar(
            "SELECT COUNT(*) FROM audit_events "
            "WHERE guild_id = ? AND action = ? AND at >= ?",
            (guild_id, action, since),
        )
        for name, action in AUDIT_ACTIONS
    }
    counts["note"] = (
        "Counted from the audit log, which only exists from logging_since "
        "onward and only covers kicks, prunes, bans, unbans and automod blocks."
    )
    return counts


def _retention_bracket(guild_id: int, logging_since: Optional[str]) -> Dict[str, Any]:
    """Did the joiners of each week come back in days 7-14?

    Bracket retention rather than N-day: Amplitude measured N-day reporting 8%
    where bracket reported 23% on the same data, and Discord membership is about
    as episodic as usage gets — nobody opens a hobby server every single day.

    Cohorts younger than 14 days are excluded outright rather than counted as
    churned; they are right-censored, and counting them drags the number down by
    however recently the server grew.
    """
    if not logging_since:
        return {
            "available": False,
            "reason": "no logging_since, so no member has a trustworthy join date",
        }
    rows = db.query(
        "SELECT strftime('%Y-W%W', mm.joined_at) AS week, COUNT(*) AS cohort, "
        "       SUM(CASE WHEN EXISTS ("
        "         SELECT 1 FROM messages x WHERE x.guild_id = mm.guild_id "
        "           AND x.author_id = mm.user_id AND x.type IN " + HUMAN_TYPES + " "
        "           AND julianday(x.created_at) BETWEEN julianday(mm.joined_at) + 7 "
        "                                           AND julianday(mm.joined_at) + 14"
        "       ) THEN 1 ELSE 0 END) AS returned "
        "FROM members mm "
        "WHERE mm.guild_id = ? AND mm.is_bot = 0 AND mm.joined_at IS NOT NULL "
        "  AND mm.joined_at >= ? "
        "  AND julianday(mm.joined_at) <= julianday('now') - 14 "
        "GROUP BY week ORDER BY week DESC",
        (guild_id, logging_since),
    )
    cohort = sum(r["cohort"] for r in rows)
    returned = sum(r["returned"] or 0 for r in rows)
    out: Dict[str, Any] = dict(stats.render_rate(returned, cohort))
    out["weeks"] = _capped(
        [
            {"week": r["week"], "joined": r["cohort"], "returned": r["returned"] or 0}
            for r in rows[:SAMPLE_COHORTS]
        ],
        SAMPLE_COHORTS,
        total=len(rows),
    )
    out["definition"] = (
        "count = joiners who posted at any point in days 7-14 after joining. "
        "Cohorts less than 14 days old are excluded, not counted as churned."
    )
    return out


def _lurker_conversion(guild_id: int, logging_since: Optional[str]) -> Dict[str, Any]:
    """Of the members who broke silence once, how many came back another day?

    Restricted to first messages at least 30 days old so every member in the
    denominator has had a fair chance, and to first messages after
    logging_since, because an earlier first message is unobservable and would
    make a long-standing regular look like a one-post lurker.
    """
    if not logging_since:
        return {
            "available": False,
            "reason": "no logging_since, so a member's first message cannot be identified",
        }
    rows = db.query(
        "SELECT f.author_id, f.first_day, ("
        "  SELECT 1 FROM messages x LEFT JOIN members xm "
        "    ON xm.guild_id = x.guild_id AND xm.user_id = x.author_id "
        "  WHERE x.guild_id = ? AND x.author_id = f.author_id "
        "    AND x.type IN " + HUMAN_TYPES + " "
        "    AND substr(x.created_at, 1, 10) > f.first_day LIMIT 1"
        ") AS came_back "
        "FROM (SELECT m.author_id, MIN(m.created_at) AS first_at, "
        "             substr(MIN(m.created_at), 1, 10) AS first_day "
        "      FROM messages m LEFT JOIN members mem "
        "        ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "      WHERE m.guild_id = ? AND COALESCE(mem.is_bot, 0) = 0 "
        "        AND m.type IN " + HUMAN_TYPES + " "
        "      GROUP BY m.author_id) f "
        "WHERE f.first_at >= ? AND julianday(f.first_at) <= julianday('now') - 30",
        (guild_id, guild_id, logging_since),
    )
    converted = sum(1 for r in rows if r["came_back"])
    out: Dict[str, Any] = dict(stats.render_rate(converted, len(rows)))
    out["definition"] = (
        "count = members whose first-ever logged message is at least 30 days old "
        "and who posted again on a later calendar day."
    )
    return out


NOT_MEASURABLE = {
    "discord_visitors": None,
    "discord_retention": None,
    "reason": "Both require client-side channel-view telemetry, which is exposed "
    "to no bot API at any permission level. Discord Server Insights is "
    "dashboard-only — the VIEW_GUILD_INSIGHTS permission bit (1<<19) exists but "
    "no REST route consumes it, and the proposal for GET /guilds/{id}/analytics "
    "was never implemented.",
}


# The exact dotted paths the self-evaluation scorer may grade. A model allowed
# to name a metric freely will, at grading time, name one that happens to have
# moved in the direction it wants.
SCOREABLE_METRICS: Tuple[str, ...] = (
    "activity.messages_7d",
    "activity.messages_30d",
    "activity.unique_posters_7d",
    "activity.days_since_owner_posted",
    "members.humans",
    "members.never_posted.count",
    "members.gone_quiet.count",
    "response_rate.answered",
    "response_rate.asked",
    "communicators_30d",
    "bus_factor_30d.factor",
    "bus_factor_30d.contributors",
    "top_share_30d.top1.share",
    "top_share_30d.owner.share",
    "membership_flow_30d.joins",
    "membership_flow_30d.leaves",
    "voice_30d.unique_participants",
    "threads.opened_30d",
    "structure.mentions_30d",
    "retention_bracket.count",
    "lurker_conversion.count",
    "dead_channels.count",
)


def resolve_metric(snap: Dict[str, Any], path: str) -> Optional[float]:
    """Walk a dotted SCOREABLE_METRICS path. None for a miss or a non-number."""
    node: Any = snap
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        return None
    return float(node)


def build(guild_id: Optional[int] = None, owner_id: Optional[int] = None) -> Dict[str, Any]:
    """Everything cadybot knows about one server. Never spans servers."""
    guild_id = guild_id or config.GUILD_ID
    if guild_id is None:
        raise SystemExit("No server chosen. Pass --guild, or set GUILD_ID in .env.")
    # The owner is whoever ran /private in this server. Falls back to 0, which
    # matches nobody — so their messages simply aren't excluded from the
    # unanswered-question check.
    owner_id = owner_id or room.owner_id(guild_id) or 0

    members = db.query(
        "SELECT user_id, username, display_name, is_bot, joined_at, invite_code, pending "
        "FROM members WHERE guild_id=? AND left_at IS NULL",
        (guild_id,),
    )
    humans = [m for m in members if not m["is_bot"]]
    # A pending member has not cleared onboarding and has therefore never seen a
    # channel, so they cannot participate in anything a stage gate would unlock.
    # Discord leaves pending False on every non-Community server, so this
    # subtraction changes nothing until the server actually has onboarding.
    pending = sum(1 for m in humans if m["pending"])
    gating_humans = len(humans) - pending

    last_post = {
        r["author_id"]: r["last_at"]
        for r in db.query(
            "SELECT author_id, MAX(created_at) AS last_at FROM messages "
            "WHERE guild_id=? AND type IN " + HUMAN_TYPES + " GROUP BY author_id",
            (guild_id,),
        )
    }

    never_posted = [_name(m) for m in humans if m["user_id"] not in last_post]
    gone_quiet = []
    for m in humans:
        age = _age_days(last_post.get(m["user_id"]))
        if age is not None and age >= config.QUIET_DAYS:
            gone_quiet.append({"member": _name(m), "silent_days": age})
    # Longest-silent first, so the sample is the interesting end of the list
    # rather than an arbitrary slice.
    gone_quiet.sort(key=lambda g: g["silent_days"], reverse=True)

    # Bots are excluded from activity counts: a chatty bot is not a lively
    # server, and counting its output would hide exactly the problem we care
    # about.
    HUMAN = (
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id=? AND COALESCE(mem.is_bot,0)=0 "
        "  AND m.type IN " + HUMAN_TYPES + " AND m.created_at>=?"
    )
    msgs_7 = db.scalar("SELECT COUNT(*) " + HUMAN, (guild_id, db.days_ago(7)))
    msgs_30 = db.scalar("SELECT COUNT(*) " + HUMAN, (guild_id, db.days_ago(30)))
    posters_7 = db.scalar(
        "SELECT COUNT(DISTINCT m.author_id) " + HUMAN, (guild_id, db.days_ago(7))
    )
    posters_30 = db.scalar(
        "SELECT COUNT(DISTINCT m.author_id) " + HUMAN, (guild_id, db.days_ago(30))
    )
    owner_msgs_7 = db.scalar(
        "SELECT COUNT(*) FROM messages WHERE guild_id=? AND author_id=? "
        "AND type IN " + HUMAN_TYPES + " AND created_at>=?",
        (guild_id, owner_id, db.days_ago(7)),
    )

    # What a founder can actually merge or delete. Threads are conversations,
    # not structure -- they have their own block -- and a category is a folder.
    # Counting them here produced "you have 14 channels, prune them" on a server
    # with one real text channel and twelve threads inside it, which is advice to
    # delete the only conversations the server has ever had.
    #
    # cadybot's own private channel is not part of the server's life either;
    # counting its briefs as activity would let it flatter itself.
    private_id = room.stored_id(guild_id) or -1
    channel_rows = db.query(
        "SELECT c.name, c.kind, c.archived, "
        "       COUNT(m.message_id) AS msgs, "
        "       COUNT(DISTINCT m.author_id) AS posters, "
        "       MAX(m.created_at) AS last_at "
        "FROM channels c "
        "LEFT JOIN messages m "
        "  ON m.guild_id = c.guild_id AND m.channel_id = c.channel_id "
        "  AND m.type IN " + HUMAN_TYPES + " AND m.created_at >= ? "
        "WHERE c.guild_id = ? AND c.channel_id != ? "
        "  AND c.kind IN " + PRUNABLE_KINDS + " "
        "GROUP BY c.channel_id ORDER BY msgs DESC",
        (db.days_ago(30), guild_id, private_id),
    )
    channels = [
        {
            "channel": r["name"],
            "kind": r["kind"],
            "messages_30d": r["msgs"],
            "unique_posters_30d": r["posters"],
            "last_activity_days_ago": _age_days(r["last_at"]),
        }
        for r in channel_rows
    ]
    dead = [r["name"] for r in channel_rows if not r["msgs"]]

    joins_30 = db.scalar(
        "SELECT COUNT(*) FROM member_events WHERE guild_id=? AND event='join' AND at>=?",
        (guild_id, db.days_ago(30)),
    )
    leaves_30 = db.scalar(
        "SELECT COUNT(*) FROM member_events WHERE guild_id=? AND event='leave' AND at>=?",
        (guild_id, db.days_ago(30)),
    )
    logging_since = db.scalar(
        "SELECT MIN(first_seen) FROM guilds WHERE guild_id=?", (guild_id,), default=None
    )
    flow: Dict[str, Any] = {
        "joins": joins_30,
        "leaves": leaves_30,
        "note": "Joins and leaves only exist from logging_since onward; "
        "they cannot be backfilled from Discord.",
    }
    # Nobody leaving is not the same as nobody being likely to leave. The rule of
    # three puts a real ceiling on a rate whose observed count is zero, which is
    # the difference between "we have no churn" and "we have not seen churn yet".
    if not leaves_30 and humans:
        flow["leave_rate_95_upper_bound"] = round(stats.rule_of_three(len(humans)), 3)
        flow["leave_rate_note"] = (
            "No leave was observed in the window. With this many members that is "
            "still consistent with a monthly leave rate up to the bound above."
        )

    invites = [
        {"invite_code": r["invite_code"], "members_still_here": r["n"]}
        for r in db.query(
            "SELECT invite_code, COUNT(*) AS n FROM members "
            "WHERE guild_id=? AND left_at IS NULL AND invite_code IS NOT NULL "
            "GROUP BY invite_code ORDER BY n DESC",
            (guild_id,),
        )
    ]

    voice_30 = db.scalar(
        "SELECT COUNT(*) FROM voice_sessions WHERE guild_id=? AND joined_at>=?",
        (guild_id, db.days_ago(30)),
    )
    voice_people_30 = db.scalar(
        "SELECT COUNT(DISTINCT user_id) FROM voice_sessions WHERE guild_id=? AND joined_at>=?",
        (guild_id, db.days_ago(30)),
    )

    top_rows = db.query(
        "SELECT m.author_id, COALESCE(mem.display_name, mem.username) AS who, COUNT(*) AS n "
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id=m.guild_id AND mem.user_id=m.author_id "
        "WHERE m.guild_id=? AND m.created_at>=? AND COALESCE(mem.is_bot,0)=0 "
        "  AND m.type IN " + HUMAN_TYPES + " "
        "GROUP BY m.author_id ORDER BY n DESC",
        (guild_id, db.days_ago(30)),
    )
    top_posters = [
        {"member": r["who"] or str(r["author_id"]), "messages_30d": r["n"]}
        for r in top_rows[:10]
    ]
    # The level only, never a trend: below roughly twenty contributors the value
    # moves week to week for reasons that have nothing to do with the server.
    contributor_counts = [r["n"] for r in top_rows]

    # Discord's own definition of a communicator: sent at least one message in a
    # text channel, or spoke for at least one second in voice. cadybot holds both
    # intents, so both halves are available.
    communicators = db.scalar(
        "SELECT COUNT(*) FROM ("
        "  SELECT m.author_id AS uid " + HUMAN +
        "  UNION SELECT user_id FROM voice_sessions WHERE guild_id=? AND joined_at>=?"
        ")",
        (guild_id, db.days_ago(30), guild_id, db.days_ago(30)),
    )
    communicator_rate = dict(stats.render_rate(communicators, len(humans)))
    communicator_rate["discord_benchmark"] = 0.30
    communicator_rate["note"] = (
        "Discord's 30% benchmark divides by VISITORS, not members. These "
        "denominators differ; do not compare them."
    )

    recent_recs = [
        {
            "days_ago": _age_days(r["created_at"]),
            "headline": r["headline"],
            "metric_to_watch": r["metric"],
            "outcome": r["outcome"],
        }
        for r in db.query(
            "SELECT created_at, headline, metric, outcome FROM recommendations "
            "WHERE guild_id=? ORDER BY created_at DESC LIMIT 8",
            (guild_id,),
        )
    ]

    members_block: Dict[str, Any] = {
        "humans": len(humans),
        "bots": len(members) - len(humans),
        "never_posted": _capped(never_posted, SAMPLE_MEMBERS),
        "never_posted_rate": stats.render_rate(len(never_posted), len(humans)),
        "gone_quiet": _capped(gone_quiet, SAMPLE_MEMBERS),
        "gone_quiet_rate": stats.render_rate(len(gone_quiet), len(humans)),
    }
    # The flat pct keys predate render_rate and other modules still read them.
    # They exist only where a percentage is defensible at all, so the floor holds
    # for them too.
    if len(humans) >= stats.MIN_RATE_DENOMINATOR:
        members_block["never_posted_pct"] = round(100 * len(never_posted) / len(humans))
        members_block["gone_quiet_pct"] = round(100 * len(gone_quiet) / len(humans))

    # Ordered deliberately. Recall of a long prompt is U-shaped in position and
    # sags by well over 30% in the middle, so the two things that must never be
    # missed sit at the ends: the unanswered questions playbooks/seed.md says
    # outrank everything else, and the explicit list of what cannot be measured,
    # which is what the model refuses with instead of inventing a figure.
    # tests/harness.py digs by key path, so nothing depends on this order.
    # Built as pairs rather than as a literal so a gated block can drop out
    # without disturbing the position of everything after it.
    ordered: List[Tuple[str, Any]] = [
        ("unanswered_questions", unanswered_questions(guild_id, owner_id)),
        ("stage", _stage(gating_humans)),
        ("generated_at", db.now()),
        ("logging_since", logging_since),
        ("members", members_block),
        (
            "activity",
            {
                "messages_7d": msgs_7,
                "messages_30d": msgs_30,
                "unique_posters_7d": posters_7,
                "unique_posters_30d": posters_30,
                "owner_messages_7d": owner_msgs_7,
                "owner_share_of_messages_7d": (
                    round(owner_msgs_7 / msgs_7, 2) if msgs_7 else None
                ),
                "days_since_owner_posted": _age_days(last_post.get(owner_id)),
            },
        ),
        ("response_rate", _response_rate(guild_id, owner_id)),
        ("communicators_30d", communicators),
        ("communicator_rate_member_denominated", communicator_rate),
        ("active_days_l28", _active_days(guild_id, len(humans))),
        (
            "bus_factor_30d",
            {
                "contributors": len(contributor_counts),
                "factor": stats.bus_factor(contributor_counts),
                "note": "Contributor Absence Factor: how few people produce half "
                "the messages. Report the level, never a trend — below roughly "
                "twenty contributors it moves week to week on its own.",
            },
        ),
        ("top_share_30d", _top_share(guild_id, owner_id)),
        ("top_posters_30d", top_posters),
        ("channels", _capped(channels, SAMPLE_CHANNELS)),
        ("dead_channels", _capped(dead, SAMPLE_CHANNELS)),
        ("threads", _threads(guild_id)),
        ("structure", _structure(guild_id, owner_id, posters_30)),
        ("membership_flow_30d", flow),
        ("moderation_30d", _moderation(guild_id)),
        ("retention_bracket", _retention_bracket(guild_id, logging_since)),
        ("lurker_conversion", _lurker_conversion(guild_id, logging_since)),
        ("invite_attribution", _capped(invites, SAMPLE_CHANNELS)),
        ("voice_30d", {"sessions": voice_30, "unique_participants": voice_people_30}),
        ("server_config", _server_config(guild_id, gating_humans)),
        ("past_recommendations", recent_recs),
        ("not_measurable", NOT_MEASURABLE),
    ]
    # top_share_30d is the only gated top-level block: when the server has not
    # produced enough conversation to divide up it is absent, not zeroed. Every
    # other key stays even when its value is null, because a null there is
    # itself the finding.
    return {
        key: value
        for key, value in ordered
        if not (value is None and key in GATED_BLOCKS)
    }
