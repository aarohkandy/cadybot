"""Deterministic server state. No LLM touches this file.

Claude never computes a number — it only explains and prescribes on top of what
this module produces. That is what stops it inventing statistics.

The metrics here are chosen to be meaningful at single-digit member counts.
Retention cohorts and DAU/MAU are deliberately absent: at n=7 they are noise.
`stage` gates which advice is even allowed downstream.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config, db, room


def _stage(humans: int) -> str:
    if humans < 25:
        return "seed"
    if humans < 100:
        return "sprout"
    if humans < 500:
        return "growing"
    return "community"


def _age_days(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        then = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - then).total_seconds() / 86400, 1)


def _name(row: Any) -> str:
    return row["display_name"] or row["username"] or str(row["user_id"])


def unanswered_questions(guild_id: int, owner_id: int) -> List[Dict[str, Any]]:
    """Member questions nobody replied to.

    A question is a non-owner, non-bot message containing '?' with no later
    message from anyone else in the same channel within UNANSWERED_HOURS.
    Crude, and correct often enough to be the single most actionable alert at
    this scale.
    """
    rows = db.query(
        """
        SELECT m.message_id, m.channel_id, m.author_id, m.created_at, m.content,
               c.name AS channel_name,
               COALESCE(mem.display_name, mem.username) AS author_name
        FROM messages m
        LEFT JOIN channels c ON c.guild_id = m.guild_id AND c.channel_id = m.channel_id
        LEFT JOIN members mem ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id
        WHERE m.guild_id = ?
          AND m.author_id != ?
          AND COALESCE(mem.is_bot, 0) = 0
          AND m.content LIKE '%?%'
          AND m.created_at >= ?
          AND NOT EXISTS (
              SELECT 1 FROM messages r
              WHERE r.guild_id = m.guild_id
                AND r.channel_id = m.channel_id
                AND r.author_id != m.author_id
                AND r.created_at > m.created_at
          )
        ORDER BY m.created_at DESC
        LIMIT 10
        """,
        (guild_id, owner_id, db.days_ago(30)),
    )
    out = []
    for r in rows:
        if _age_days(r["created_at"]) is None:
            continue
        hours_old = (_age_days(r["created_at"]) or 0) * 24
        if hours_old < config.UNANSWERED_HOURS:
            continue  # still fresh; not yet a failure
        out.append(
            {
                "author": r["author_name"] or str(r["author_id"]),
                "channel": r["channel_name"],
                "asked_days_ago": _age_days(r["created_at"]),
                "text": (r["content"] or "")[:400],
                "link": "https://discord.com/channels/%d/%d/%d"
                % (guild_id, r["channel_id"], r["message_id"]),
            }
        )
    return out


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
        "SELECT user_id, username, display_name, is_bot, joined_at, invite_code "
        "FROM members WHERE guild_id=? AND left_at IS NULL",
        (guild_id,),
    )
    humans = [m for m in members if not m["is_bot"]]

    last_post = {
        r["author_id"]: r["last_at"]
        for r in db.query(
            "SELECT author_id, MAX(created_at) AS last_at FROM messages "
            "WHERE guild_id=? GROUP BY author_id",
            (guild_id,),
        )
    }

    never_posted = [_name(m) for m in humans if m["user_id"] not in last_post]
    gone_quiet = []
    for m in humans:
        age = _age_days(last_post.get(m["user_id"]))
        if age is not None and age >= config.QUIET_DAYS:
            gone_quiet.append({"member": _name(m), "silent_days": age})

    # Bots are excluded from activity counts: a chatty bot is not a lively
    # server, and counting its output would hide exactly the problem we care
    # about.
    HUMAN = (
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id=? AND COALESCE(mem.is_bot,0)=0 AND m.created_at>=?"
    )
    msgs_7 = db.scalar("SELECT COUNT(*) " + HUMAN, (guild_id, db.days_ago(7)))
    msgs_30 = db.scalar("SELECT COUNT(*) " + HUMAN, (guild_id, db.days_ago(30)))
    posters_7 = db.scalar(
        "SELECT COUNT(DISTINCT m.author_id) " + HUMAN, (guild_id, db.days_ago(7))
    )
    owner_msgs_7 = db.scalar(
        "SELECT COUNT(*) FROM messages WHERE guild_id=? AND author_id=? AND created_at>=?",
        (guild_id, owner_id, db.days_ago(7)),
    )

    # cadybot's own private channel is not part of the server's life; counting
    # its briefs as activity would let it flatter itself.
    private_id = room.stored_id(guild_id) or -1
    channels = [
        {
            "channel": r["name"],
            "kind": r["kind"],
            "messages_30d": r["msgs"],
            "unique_posters_30d": r["posters"],
            "last_activity_days_ago": _age_days(r["last_at"]),
        }
        for r in db.query(
            """
            SELECT c.name, c.kind,
                   COUNT(m.message_id) AS msgs,
                   COUNT(DISTINCT m.author_id) AS posters,
                   MAX(m.created_at) AS last_at
            FROM channels c
            LEFT JOIN messages m
              ON m.guild_id = c.guild_id AND m.channel_id = c.channel_id
              AND m.created_at >= ?
            WHERE c.guild_id = ? AND c.channel_id != ?
            GROUP BY c.channel_id
            ORDER BY msgs DESC
            """,
            (db.days_ago(30), guild_id, private_id),
        )
    ]

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

    top_posters = [
        {"member": r["who"] or str(r["author_id"]), "messages_30d": r["n"]}
        for r in db.query(
            "SELECT m.author_id, COALESCE(mem.display_name, mem.username) AS who, COUNT(*) AS n "
            "FROM messages m LEFT JOIN members mem "
            "  ON mem.guild_id=m.guild_id AND mem.user_id=m.author_id "
            "WHERE m.guild_id=? AND m.created_at>=? AND COALESCE(mem.is_bot,0)=0 "
            "GROUP BY m.author_id ORDER BY n DESC LIMIT 10",
            (guild_id, db.days_ago(30)),
        )
    ]

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

    return {
        "generated_at": db.now(),
        "stage": _stage(len(humans)),
        "logging_since": logging_since,
        "members": {
            "humans": len(humans),
            "bots": len(members) - len(humans),
            "never_posted": never_posted,
            "gone_quiet": gone_quiet,
        },
        "activity": {
            "messages_7d": msgs_7,
            "messages_30d": msgs_30,
            "unique_posters_7d": posters_7,
            "owner_messages_7d": owner_msgs_7,
            "owner_share_of_messages_7d": (
                round(owner_msgs_7 / msgs_7, 2) if msgs_7 else None
            ),
            "days_since_owner_posted": _age_days(last_post.get(owner_id)),
        },
        "channels": channels,
        "membership_flow_30d": {
            "joins": joins_30,
            "leaves": leaves_30,
            "note": "Joins and leaves only exist from logging_since onward; "
            "they cannot be backfilled from Discord.",
        },
        "invite_attribution": invites,
        "voice_30d": {"sessions": voice_30, "unique_participants": voice_people_30},
        "top_posters_30d": top_posters,
        "unanswered_questions": unanswered_questions(guild_id, owner_id),
        "past_recommendations": recent_recs,
    }
