"""SQLite storage.

Every table carries guild_id and no query in this file crosses guilds, so the
schema can go multi-tenant later without a rewrite. Timestamps are ISO-8601 UTC
strings, which sort lexicographically — that is the only reason they are text.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS guilds (
    guild_id    INTEGER PRIMARY KEY,
    name        TEXT,
    first_seen  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channels (
    guild_id    INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    name        TEXT,
    kind        TEXT,
    parent_id   INTEGER,
    created_at  TEXT,
    PRIMARY KEY (guild_id, channel_id)
);

CREATE TABLE IF NOT EXISTS members (
    guild_id      INTEGER NOT NULL,
    user_id       INTEGER NOT NULL,
    username      TEXT,
    display_name  TEXT,
    is_bot        INTEGER NOT NULL DEFAULT 0,
    joined_at     TEXT,
    first_seen    TEXT NOT NULL,
    left_at       TEXT,
    invite_code   TEXT,
    PRIMARY KEY (guild_id, user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    guild_id     INTEGER NOT NULL,
    channel_id   INTEGER NOT NULL,
    message_id   INTEGER NOT NULL,
    author_id    INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    content      TEXT,
    reply_to_id  INTEGER,
    attachments  INTEGER NOT NULL DEFAULT 0,
    reactions    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, message_id)
);
CREATE INDEX IF NOT EXISTS ix_messages_time    ON messages (guild_id, created_at);
CREATE INDEX IF NOT EXISTS ix_messages_channel ON messages (guild_id, channel_id, created_at);
CREATE INDEX IF NOT EXISTS ix_messages_author  ON messages (guild_id, author_id, created_at);

CREATE TABLE IF NOT EXISTS member_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id  INTEGER NOT NULL,
    user_id   INTEGER NOT NULL,
    event     TEXT NOT NULL,          -- join | leave
    at        TEXT NOT NULL,
    invite_code TEXT
);
CREATE INDEX IF NOT EXISTS ix_member_events ON member_events (guild_id, at);

CREATE TABLE IF NOT EXISTS voice_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    joined_at   TEXT NOT NULL,
    left_at     TEXT
);
CREATE INDEX IF NOT EXISTS ix_voice ON voice_sessions (guild_id, joined_at);

-- Snapshot of invite use-counts, so a join can be attributed to the invite that
-- produced it by diffing against the previous snapshot.
CREATE TABLE IF NOT EXISTS invite_uses (
    guild_id  INTEGER NOT NULL,
    code      TEXT NOT NULL,
    uses      INTEGER NOT NULL,
    inviter_id INTEGER,
    seen_at   TEXT NOT NULL,
    PRIMARY KEY (guild_id, code)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id      INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    headline      TEXT NOT NULL,
    action        TEXT NOT NULL,
    evidence      TEXT,
    metric        TEXT,
    prediction    TEXT,
    reviewed_at   TEXT,
    outcome       TEXT
);
CREATE INDEX IF NOT EXISTS ix_recs ON recommendations (guild_id, created_at);

CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id       INTEGER NOT NULL,
    kind           TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    model          TEXT,
    input_tokens   INTEGER,
    output_tokens  INTEGER,
    cache_read     INTEGER
);

-- Conversation history for the private channel, so talking to cadybot survives
-- a restart. Kept per channel and trimmed to the last few turns.
CREATE TABLE IF NOT EXISTS conversation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    channel_id  INTEGER NOT NULL,
    role        TEXT NOT NULL,          -- user | assistant
    speaker     TEXT,
    content     TEXT NOT NULL,
    at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_conversation ON conversation (guild_id, channel_id, id);

CREATE TABLE IF NOT EXISTS settings (
    guild_id  INTEGER NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT,
    PRIMARY KEY (guild_id, key)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def hours_ago(hours: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


_conn: Optional[sqlite3.Connection] = None


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(str(config.DB_PATH), isolation_level=None)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.executescript(SCHEMA)
    return _conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = connect()
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def query(sql: str, params: tuple = ()) -> List[sqlite3.Row]:
    return connect().execute(sql, params).fetchall()


def one(sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    return connect().execute(sql, params).fetchone()


def scalar(sql: str, params: tuple = (), default: Any = 0) -> Any:
    row = one(sql, params)
    if row is None or row[0] is None:
        return default
    return row[0]


# --- writes ----------------------------------------------------------------


def upsert_guild(guild_id: int, name: str) -> None:
    connect().execute(
        "INSERT INTO guilds (guild_id, name, first_seen) VALUES (?, ?, ?) "
        "ON CONFLICT(guild_id) DO UPDATE SET name=excluded.name",
        (guild_id, name, now()),
    )


def upsert_channel(
    guild_id: int,
    channel_id: int,
    name: Optional[str],
    kind: str,
    parent_id: Optional[int],
    created_at: Optional[str],
) -> None:
    connect().execute(
        "INSERT INTO channels (guild_id, channel_id, name, kind, parent_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(guild_id, channel_id) DO UPDATE SET "
        "name=excluded.name, kind=excluded.kind, parent_id=excluded.parent_id",
        (guild_id, channel_id, name, kind, parent_id, created_at),
    )


def upsert_member(
    guild_id: int,
    user_id: int,
    username: Optional[str],
    display_name: Optional[str],
    is_bot: bool,
    joined_at: Optional[str],
) -> None:
    connect().execute(
        "INSERT INTO members (guild_id, user_id, username, display_name, is_bot, joined_at, first_seen) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(guild_id, user_id) DO UPDATE SET "
        "username=excluded.username, display_name=excluded.display_name, "
        "is_bot=excluded.is_bot, joined_at=COALESCE(members.joined_at, excluded.joined_at), "
        "left_at=NULL",
        (guild_id, user_id, username, display_name, 1 if is_bot else 0, joined_at, now()),
    )


def mark_left(guild_id: int, user_id: int) -> None:
    connect().execute(
        "UPDATE members SET left_at=? WHERE guild_id=? AND user_id=?",
        (now(), guild_id, user_id),
    )


def record_member_event(
    guild_id: int, user_id: int, event: str, invite_code: Optional[str] = None
) -> None:
    connect().execute(
        "INSERT INTO member_events (guild_id, user_id, event, at, invite_code) VALUES (?, ?, ?, ?, ?)",
        (guild_id, user_id, event, now(), invite_code),
    )


def upsert_message(row: Dict[str, Any]) -> None:
    connect().execute(
        "INSERT INTO messages (guild_id, channel_id, message_id, author_id, created_at, "
        "content, reply_to_id, attachments, reactions) "
        "VALUES (:guild_id, :channel_id, :message_id, :author_id, :created_at, "
        ":content, :reply_to_id, :attachments, :reactions) "
        "ON CONFLICT(guild_id, message_id) DO UPDATE SET "
        "content=excluded.content, reactions=excluded.reactions",
        row,
    )


def bump_reactions(guild_id: int, message_id: int, delta: int) -> None:
    connect().execute(
        "UPDATE messages SET reactions = MAX(0, reactions + ?) WHERE guild_id=? AND message_id=?",
        (delta, guild_id, message_id),
    )


def open_voice(guild_id: int, channel_id: int, user_id: int) -> None:
    connect().execute(
        "INSERT INTO voice_sessions (guild_id, channel_id, user_id, joined_at) VALUES (?, ?, ?, ?)",
        (guild_id, channel_id, user_id, now()),
    )


def close_voice(guild_id: int, user_id: int) -> None:
    connect().execute(
        "UPDATE voice_sessions SET left_at=? "
        "WHERE id = (SELECT id FROM voice_sessions WHERE guild_id=? AND user_id=? "
        "AND left_at IS NULL ORDER BY joined_at DESC LIMIT 1)",
        (now(), guild_id, user_id),
    )


def snapshot_invites(guild_id: int, invites: List[Dict[str, Any]]) -> Dict[str, int]:
    """Store current invite use-counts, returning the previous counts."""
    previous = {
        r["code"]: r["uses"]
        for r in query("SELECT code, uses FROM invite_uses WHERE guild_id=?", (guild_id,))
    }
    stamp = now()
    for inv in invites:
        connect().execute(
            "INSERT INTO invite_uses (guild_id, code, uses, inviter_id, seen_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(guild_id, code) DO UPDATE SET uses=excluded.uses, seen_at=excluded.seen_at",
            (guild_id, inv["code"], inv["uses"], inv.get("inviter_id"), stamp),
        )
    return previous


def attribute_invite(guild_id: int, user_id: int, code: Optional[str]) -> None:
    if not code:
        return
    connect().execute(
        "UPDATE members SET invite_code=? WHERE guild_id=? AND user_id=?",
        (code, guild_id, user_id),
    )


def save_recommendations(guild_id: int, recs: List[Dict[str, Any]]) -> None:
    stamp = now()
    for r in recs:
        connect().execute(
            "INSERT INTO recommendations (guild_id, created_at, headline, action, evidence, metric, prediction) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                guild_id,
                stamp,
                r.get("headline", ""),
                r.get("action", ""),
                r.get("evidence", ""),
                r.get("metric", ""),
                r.get("prediction", ""),
            ),
        )


def record_run(guild_id: int, kind: str, usage: Any, model: str) -> None:
    connect().execute(
        "INSERT INTO runs (guild_id, kind, created_at, model, input_tokens, output_tokens, cache_read) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            guild_id,
            kind,
            now(),
            model,
            getattr(usage, "input_tokens", None),
            getattr(usage, "output_tokens", None),
            getattr(usage, "cache_read_input_tokens", None),
        ),
    )


def record_local_run(
    guild_id: int, kind: str, model: str, prompt_tokens: int, output_tokens: int
) -> None:
    connect().execute(
        "INSERT INTO runs (guild_id, kind, created_at, model, input_tokens, output_tokens, cache_read) "
        "VALUES (?, ?, ?, ?, ?, ?, 0)",
        (guild_id, kind, now(), model, prompt_tokens, output_tokens),
    )


CONVERSATION_TURNS = 16


def add_turn(
    guild_id: int, channel_id: int, role: str, content: str, speaker: Optional[str] = None
) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO conversation (guild_id, channel_id, role, speaker, content, at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (guild_id, channel_id, role, speaker, content, now()),
    )
    # Keep only the most recent turns; old context stops being useful long
    # before it stops costing tokens.
    conn.execute(
        "DELETE FROM conversation WHERE guild_id=? AND channel_id=? AND id NOT IN "
        "(SELECT id FROM conversation WHERE guild_id=? AND channel_id=? "
        " ORDER BY id DESC LIMIT ?)",
        (guild_id, channel_id, guild_id, channel_id, CONVERSATION_TURNS),
    )


def recent_turns(guild_id: int, channel_id: int) -> List[Dict[str, str]]:
    rows = query(
        "SELECT role, speaker, content FROM conversation "
        "WHERE guild_id=? AND channel_id=? ORDER BY id",
        (guild_id, channel_id),
    )
    out = []
    for r in rows:
        text = r["content"]
        if r["role"] == "user" and r["speaker"]:
            text = "%s: %s" % (r["speaker"], text)
        out.append({"role": r["role"], "content": text})
    return out


def clear_turns(guild_id: int, channel_id: int) -> int:
    return connect().execute(
        "DELETE FROM conversation WHERE guild_id=? AND channel_id=?", (guild_id, channel_id)
    ).rowcount


def get_setting(guild_id: int, key: str) -> Optional[str]:
    row = one("SELECT value FROM settings WHERE guild_id=? AND key=?", (guild_id, key))
    return row["value"] if row else None


def set_setting(guild_id: int, key: str, value: Optional[str]) -> None:
    connect().execute(
        "INSERT INTO settings (guild_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(guild_id, key) DO UPDATE SET value=excluded.value",
        (guild_id, key, value),
    )


def purge_member(guild_id: int, user_id: int) -> int:
    """Delete everything about one member. Data-deletion requests land here."""
    conn = connect()
    removed = 0
    for sql in (
        "DELETE FROM messages WHERE guild_id=? AND author_id=?",
        "DELETE FROM member_events WHERE guild_id=? AND user_id=?",
        "DELETE FROM voice_sessions WHERE guild_id=? AND user_id=?",
        "DELETE FROM members WHERE guild_id=? AND user_id=?",
    ):
        removed += conn.execute(sql, (guild_id, user_id)).rowcount
    return removed
