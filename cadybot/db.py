"""SQLite storage.

Every table carries guild_id and no query in this file crosses guilds, so the
schema can go multi-tenant later without a rewrite. Timestamps are ISO-8601 UTC
strings, which sort lexicographically — that is the only reason they are text.
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS guilds (
    guild_id    INTEGER PRIMARY KEY,
    name        TEXT,
    first_seen  TEXT NOT NULL
);

-- The thread columns are NULL for anything that is not a Thread. They are what
-- makes "ten of twelve threads auto-archived at 1440 minutes with under four
-- messages" answerable without re-walking Discord.
CREATE TABLE IF NOT EXISTS channels (
    guild_id               INTEGER NOT NULL,
    channel_id             INTEGER NOT NULL,
    name                   TEXT,
    kind                   TEXT,
    parent_id              INTEGER,
    created_at             TEXT,
    archived               INTEGER,
    archive_timestamp      TEXT,
    auto_archive_duration  INTEGER,
    thread_message_count   INTEGER,
    parent_kind            TEXT,
    PRIMARY KEY (guild_id, channel_id)
);

CREATE TABLE IF NOT EXISTS members (
    guild_id         INTEGER NOT NULL,
    user_id          INTEGER NOT NULL,
    username         TEXT,
    display_name     TEXT,
    is_bot           INTEGER NOT NULL DEFAULT 0,
    joined_at        TEXT,
    first_seen       TEXT NOT NULL,
    left_at          TEXT,
    invite_code      TEXT,
    flags            INTEGER,
    pending          INTEGER NOT NULL DEFAULT 0,
    premium_since    TEXT,
    timed_out_until  TEXT,
    PRIMARY KEY (guild_id, user_id)
);

-- `type` is Discord's MessageType: 0 is an ordinary message, 19 a reply, 21 the
-- content-free mirror Discord posts in the parent channel when a thread starts.
-- `ref_type` splits the one `reference` field Discord overloads: NULL for no
-- reference, 0 for a genuine reply, 1 for a forward. A forward's text was
-- authored in a server cadybot is not in and is never stored.
CREATE TABLE IF NOT EXISTS messages (
    guild_id          INTEGER NOT NULL,
    channel_id        INTEGER NOT NULL,
    message_id        INTEGER NOT NULL,
    author_id         INTEGER NOT NULL,
    created_at        TEXT NOT NULL,
    content           TEXT,
    reply_to_id       INTEGER,
    attachments       INTEGER NOT NULL DEFAULT 0,
    reactions         INTEGER NOT NULL DEFAULT 0,
    type              INTEGER NOT NULL DEFAULT 0,
    ref_type          INTEGER,
    flags             INTEGER NOT NULL DEFAULT 0,
    edited_at         TEXT,
    mention_everyone  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, message_id)
);
CREATE INDEX IF NOT EXISTS ix_messages_time    ON messages (guild_id, created_at);
CREATE INDEX IF NOT EXISTS ix_messages_channel ON messages (guild_id, channel_id, created_at);
CREATE INDEX IF NOT EXISTS ix_messages_author  ON messages (guild_id, author_id, created_at);

-- IDs only, never text. Populated from the gateway's mentions array rather than
-- by scanning content, so it does not depend on the Message Content intent.
-- @everyone and @here are not user mentions and live on messages instead.
CREATE TABLE IF NOT EXISTS mentions (
    guild_id      INTEGER NOT NULL,
    message_id    INTEGER NOT NULL,
    author_id     INTEGER NOT NULL,
    mentioned_id  INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    PRIMARY KEY (guild_id, message_id, mentioned_id)
);
CREATE INDEX IF NOT EXISTS ix_mentions_time ON mentions (guild_id, created_at);

-- One row per person per emoji, so messages.reactions can be recomputed rather
-- than incremented. A counter moved by deltas drifts upward forever the first
-- time a remove or clear event is missed.
CREATE TABLE IF NOT EXISTS reactions (
    guild_id    INTEGER NOT NULL,
    message_id  INTEGER NOT NULL,
    user_id     INTEGER NOT NULL,
    emoji       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (guild_id, message_id, user_id, emoji)
);
CREATE INDEX IF NOT EXISTS ix_reactions_msg ON reactions (guild_id, message_id);

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

-- Written only by scorecard.pre_register, which owns the extra columns it adds
-- to this table. There is deliberately no insert helper here: a row stored
-- without its baseline, its threshold and its guardrail floor can never be
-- graded, and a second writer is a way to open one that never will be.
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
    outcome       TEXT,
    play          TEXT                    -- which catalogue play this was
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

-- Server configuration cadybot can observe. NULL means NOT READABLE and must
-- stay distinguishable from 0: reporting "onboarding is off" when the truth is
-- "cadybot cannot see onboarding" sends the founder to fix a setting that is
-- already correct.
CREATE TABLE IF NOT EXISTS guild_facts (
    guild_id                     INTEGER PRIMARY KEY,
    at                           TEXT NOT NULL,
    onboarding_enabled           INTEGER,
    onboarding_mode              TEXT,
    onboarding_prompts           INTEGER,
    onboarding_required_prompts  INTEGER,
    onboarding_default_channels  INTEGER,
    widget_enabled               INTEGER,
    boost_count                  INTEGER,
    boost_tier                   INTEGER,
    verification_level           TEXT,
    is_community                 INTEGER,
    audit_readable               INTEGER,
    onboarding_readable          INTEGER
);

-- The only signal cadybot gets about members who are present but never speak.
-- Both counts come from a fetched guild; the cached object leaves them unset.
CREATE TABLE IF NOT EXISTS presence_samples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id        INTEGER NOT NULL,
    at              TEXT NOT NULL,
    approx_members  INTEGER,
    approx_online   INTEGER
);
CREATE INDEX IF NOT EXISTS ix_presence ON presence_samples (guild_id, at);

-- A narrow moderation slice of the audit log. Kicks, bans and prunes are the
-- only leaves Discord will ever tell us the reason for; everything else in the
-- audit log is server administration and says nothing about community health.
CREATE TABLE IF NOT EXISTS audit_events (
    guild_id    INTEGER NOT NULL,
    entry_id    INTEGER NOT NULL,
    action      INTEGER NOT NULL,
    user_id     INTEGER,
    target_id   INTEGER,
    at          TEXT NOT NULL,
    extra_json  TEXT,
    PRIMARY KEY (guild_id, entry_id)
);
CREATE INDEX IF NOT EXISTS ix_audit ON audit_events (guild_id, at);

-- A daily close of each event-count metric: the first history of past readings
-- this database has ever held. snapshot.build recomputes from the raw tables
-- and discards, so "what did this number look like a fortnight ago" had no
-- answer. Wall-clock ages and stock counts are deliberately absent — a metric
-- that moves because time passed is a clock, and a detector watching one fires
-- forever on a server where nothing is happening.
CREATE TABLE IF NOT EXISTS ledger (
    guild_id  INTEGER NOT NULL,
    day       TEXT NOT NULL,         -- 'YYYY-MM-DD', UTC
    metric    TEXT NOT NULL,         -- a ledger.LEDGER_METRICS path
    value     REAL,
    PRIMARY KEY (guild_id, day, metric)
);

-- One row per thought, written BEFORE the model call. That order makes the
-- insert do four jobs at once: it charges the budget, advances the cursor,
-- survives a crash mid-call, and locks against a second process. `provoked_by`
-- is copied from a stored row, never generated, so two processes racing on the
-- same event compute byte-identical values and the loser's INSERT loses to the
-- UNIQUE constraint instead of duplicating the thought.
CREATE TABLE IF NOT EXISTS journal (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id      INTEGER NOT NULL,
    kind          TEXT NOT NULL,     -- verdict | life | joined | context | drift
    provoked_by   TEXT NOT NULL,     -- a stored timestamp. NEVER db.now().
    about_ref     TEXT,              -- 'R-14', or NULL
    started_at    TEXT NOT NULL,
    self_prompt   TEXT NOT NULL,     -- composed by code, stored verbatim
    restated      TEXT,
    reasoning     TEXT,
    evidence      TEXT,
    note_to_self  TEXT,
    watch_metric  TEXT,
    to_founder    TEXT,
    unverified    TEXT,              -- JSON list, reported not enforced
    outcome       TEXT NOT NULL DEFAULT 'started',   -- started | thought | failed
    failure       TEXT,
    wanted_telling INTEGER NOT NULL DEFAULT 0,        -- the model lifted its veto
    attempts      INTEGER NOT NULL DEFAULT 1,         -- a timed-out backend gets one retry
    finding       TEXT,                               -- the SQL-written headline fact
    surfaced_at   TEXT,
    model         TEXT,
    UNIQUE (guild_id, kind, provoked_by)
);
CREATE INDEX IF NOT EXISTS ix_journal ON journal (guild_id, started_at);

-- What cadybot actually said, and when. Nothing recorded this before, so the
-- nightly pass, the weekly brief and the unanswered-question alert had no way
-- to know about each other: three mouths and no shared memory of having
-- spoken. Anything that wants to reason about how often the founder is
-- interrupted needs this table to exist first.
CREATE TABLE IF NOT EXISTS deliveries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id    INTEGER NOT NULL,
    at          TEXT NOT NULL,
    kind        TEXT NOT NULL,       -- nightly | weekly | unanswered | thought | other
    chars       INTEGER,
    journal_id  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_deliveries ON deliveries (guild_id, at);
"""

# CREATE TABLE IF NOT EXISTS will not add a column to a table that already
# exists, so every column introduced after a release has to appear twice: in
# SCHEMA, so a fresh database is right, and here, so the live one catches up.
MIGRATIONS: List[Tuple[str, str, str]] = [
    ("messages", "type", "INTEGER NOT NULL DEFAULT 0"),
    ("messages", "ref_type", "INTEGER"),
    ("messages", "flags", "INTEGER NOT NULL DEFAULT 0"),
    ("messages", "edited_at", "TEXT"),
    ("messages", "mention_everyone", "INTEGER NOT NULL DEFAULT 0"),
    ("channels", "archived", "INTEGER"),
    ("channels", "archive_timestamp", "TEXT"),
    ("channels", "auto_archive_duration", "INTEGER"),
    ("channels", "thread_message_count", "INTEGER"),
    ("channels", "parent_kind", "TEXT"),
    ("members", "flags", "INTEGER"),
    ("members", "pending", "INTEGER NOT NULL DEFAULT 0"),
    ("members", "premium_since", "TEXT"),
    ("members", "timed_out_until", "TEXT"),
    # journal already exists on the live database, so CREATE TABLE IF NOT EXISTS
    # will not add these; every post-release column has to appear twice.
    ("journal", "wanted_telling", "INTEGER NOT NULL DEFAULT 0"),
    ("journal", "attempts", "INTEGER NOT NULL DEFAULT 1"),
    ("journal", "finding", "TEXT"),
    ("recommendations", "play", "TEXT"),
]

# Settings row marking the one-off phantom sweep as already done. guild_id 0 is
# not a real server, so it cannot collide with a per-guild setting.
CLEANUP_KEY = "cleanup_thread_starters"


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


# One connection per thread rather than one per process. Everything slow in
# cadybot — a model call and the database work either side of it — runs on a
# worker thread so the gateway heartbeat keeps beating, and Python's sqlite3
# refuses to serve a connection to a thread that did not create it. A single
# shared connection therefore made every /ask, /brief and private-channel reply
# die with ProgrammingError the moment it touched the database from that thread.
#
# check_same_thread=False plus one shared connection is the other way to spell
# this and is not safe here: sqlite is compiled THREADSAFE=2, which permits many
# connections but not one connection used concurrently, and gateway ingest on
# the event loop genuinely overlaps a brief being written on a worker. Separate
# connections over WAL is the arrangement sqlite supports — readers never block,
# and busy_timeout absorbs the one-writer-at-a-time window.
_local = threading.local()


def connect() -> sqlite3.Connection:
    conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(config.DB_PATH), isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        _migrate(conn)
        _local.conn = conn
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns SCHEMA declares that an older database is missing.

    Every ALTER is guarded by the PRAGMA rather than by catching the error, so a
    second run is a silent no-op and an unexpected schema cannot abort startup.
    """
    for table, column, decl in MIGRATIONS:
        present = {row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)}
        if not present:
            continue  # table does not exist here; SCHEMA above owns creating it
        if column not in present:
            conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, decl))
    _sweep_thread_starters(conn)


def _sweep_thread_starters(conn: sqlite3.Connection) -> None:
    """One-off removal of the phantom rows earlier ingest mistook for messages.

    When a thread is created from a message Discord posts a type-21 mirror in the
    parent channel. It has no content and no attachments, but it does carry a
    reference, so it was being stored as both a message and a reply. On the live
    database that was 12 of 100 rows and 12 of the 17 replies — enough to move
    every rate downstream. Ingest no longer treats those references as replies,
    which makes this signature safe to delete once.
    """
    if conn.execute(
        "SELECT 1 FROM settings WHERE guild_id=0 AND key=?", (CLEANUP_KEY,)
    ).fetchone():
        return
    doomed = (
        "SELECT guild_id, message_id FROM messages WHERE content IS NULL "
        "AND attachments = 0 AND reply_to_id IS NOT NULL"
    )
    # Cascade the same way delete_messages does. Dropping only the message rows
    # would strand their mentions, and structure.mentions_30d counts mentions
    # rows directly — it would keep reporting @-mentions belonging to messages
    # that no longer exist.
    for table in ("mentions", "reactions"):
        conn.execute(
            "DELETE FROM %s WHERE (guild_id, message_id) IN (%s)" % (table, doomed)
        )
    removed = conn.execute(
        "DELETE FROM messages WHERE content IS NULL AND attachments = 0 "
        "AND reply_to_id IS NOT NULL"
    ).rowcount
    conn.execute(
        "INSERT INTO settings (guild_id, key, value) VALUES (0, ?, ?)",
        (CLEANUP_KEY, str(removed)),
    )
    if removed:
        print("removed %d phantom thread-starter rows" % removed)


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
    archived: Optional[int] = None,
    archive_timestamp: Optional[str] = None,
    auto_archive_duration: Optional[int] = None,
    thread_message_count: Optional[int] = None,
    parent_kind: Optional[str] = None,
) -> None:
    connect().execute(
        "INSERT INTO channels (guild_id, channel_id, name, kind, parent_id, created_at, "
        "archived, archive_timestamp, auto_archive_duration, thread_message_count, parent_kind) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(guild_id, channel_id) DO UPDATE SET "
        "name=excluded.name, kind=excluded.kind, parent_id=excluded.parent_id, "
        "archived=excluded.archived, archive_timestamp=excluded.archive_timestamp, "
        "auto_archive_duration=excluded.auto_archive_duration, "
        "thread_message_count=excluded.thread_message_count, parent_kind=excluded.parent_kind",
        (
            guild_id, channel_id, name, kind, parent_id, created_at,
            archived, archive_timestamp, auto_archive_duration,
            thread_message_count, parent_kind,
        ),
    )


def member_state(member: Any) -> Optional[Dict[str, Any]]:
    """The fields only a real Member carries, or None for a bare User.

    A message author who has since left the server arrives as a User with no
    onboarding or timeout state at all. Returning None there lets upsert_member
    tell "not a member object" apart from "member with nothing set", so stale
    facts are kept instead of being blanked by an ordinary message.
    """
    if not hasattr(member, "pending") or not hasattr(member, "flags"):
        return None
    return {
        "flags": getattr(member.flags, "value", None),
        "pending": bool(member.pending),
        "premium_since": iso(getattr(member, "premium_since", None)),
        "timed_out_until": iso(getattr(member, "timed_out_until", None)),
    }


def upsert_member(
    guild_id: int,
    user_id: int,
    username: Optional[str],
    display_name: Optional[str],
    is_bot: bool,
    joined_at: Optional[str],
    state: Optional[Dict[str, Any]] = None,
) -> None:
    conn = connect()
    conn.execute(
        "INSERT INTO members (guild_id, user_id, username, display_name, is_bot, joined_at, first_seen) "
        "VALUES (?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(guild_id, user_id) DO UPDATE SET "
        "username=excluded.username, display_name=excluded.display_name, "
        "is_bot=excluded.is_bot, joined_at=COALESCE(members.joined_at, excluded.joined_at), "
        "left_at=NULL",
        (guild_id, user_id, username, display_name, 1 if is_bot else 0, joined_at, now()),
    )
    if state is None:
        return
    conn.execute(
        "UPDATE members SET flags=?, pending=?, premium_since=?, timed_out_until=? "
        "WHERE guild_id=? AND user_id=?",
        (
            state.get("flags"),
            1 if state.get("pending") else 0,
            state.get("premium_since"),
            state.get("timed_out_until"),
            guild_id,
            user_id,
        ),
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
    """Insert or refresh one message.

    Columns added after the first release may be omitted by the caller and fall
    back to the same values the schema defaults to, so a row built by an older
    caller is still a valid row.
    """
    params = {
        "guild_id": row["guild_id"],
        "channel_id": row["channel_id"],
        "message_id": row["message_id"],
        "author_id": row["author_id"],
        "created_at": row["created_at"],
        "content": row.get("content"),
        "reply_to_id": row.get("reply_to_id"),
        "attachments": row.get("attachments") or 0,
        "reactions": row.get("reactions") or 0,
        "type": row.get("type") or 0,
        "ref_type": row.get("ref_type"),
        "flags": row.get("flags") or 0,
        "edited_at": row.get("edited_at"),
        "mention_everyone": 1 if row.get("mention_everyone") else 0,
    }
    connect().execute(
        "INSERT INTO messages (guild_id, channel_id, message_id, author_id, created_at, "
        "content, reply_to_id, attachments, reactions, type, ref_type, flags, edited_at, "
        "mention_everyone) "
        "VALUES (:guild_id, :channel_id, :message_id, :author_id, :created_at, "
        ":content, :reply_to_id, :attachments, :reactions, :type, :ref_type, :flags, "
        ":edited_at, :mention_everyone) "
        "ON CONFLICT(guild_id, message_id) DO UPDATE SET "
        "content=excluded.content, reactions=excluded.reactions, "
        "edited_at=excluded.edited_at, flags=excluded.flags",
        params,
    )


def update_message_content(
    guild_id: int,
    message_id: int,
    content: Optional[str],
    edited_at: Optional[str] = None,
    flags: Optional[int] = None,
) -> None:
    connect().execute(
        "UPDATE messages SET content=?, edited_at=COALESCE(?, edited_at), "
        "flags=COALESCE(?, flags) WHERE guild_id=? AND message_id=?",
        (content or None, edited_at, flags, guild_id, message_id),
    )


def delete_messages(guild_id: int, message_ids: List[int]) -> int:
    """Hard delete, cascading to mentions and reactions.

    Someone deleting their own message is the clearest signal there is that
    keeping it is no longer warranted, so nothing is retained — not a tombstone,
    not the text, not the reactions it collected.
    """
    if not message_ids:
        return 0
    conn = connect()
    marks = ",".join("?" * len(message_ids))
    params = tuple([guild_id] + list(message_ids))
    for table in ("mentions", "reactions"):
        conn.execute(
            "DELETE FROM %s WHERE guild_id=? AND message_id IN (%s)" % (table, marks), params
        )
    return conn.execute(
        "DELETE FROM messages WHERE guild_id=? AND message_id IN (%s)" % marks, params
    ).rowcount


def add_mentions(
    guild_id: int,
    message_id: int,
    author_id: int,
    mentioned_ids: List[int],
    created_at: str,
) -> None:
    conn = connect()
    for mentioned_id in mentioned_ids:
        conn.execute(
            "INSERT OR IGNORE INTO mentions "
            "(guild_id, message_id, author_id, mentioned_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (guild_id, message_id, author_id, mentioned_id, created_at),
        )


def _recount_reactions(guild_id: int, message_id: int) -> None:
    """Derive messages.reactions from the rows rather than nudging a counter."""
    connect().execute(
        "UPDATE messages SET reactions = "
        "(SELECT COUNT(*) FROM reactions WHERE guild_id=? AND message_id=?) "
        "WHERE guild_id=? AND message_id=?",
        (guild_id, message_id, guild_id, message_id),
    )


def add_reaction(guild_id: int, message_id: int, user_id: int, emoji: str) -> None:
    connect().execute(
        "INSERT OR IGNORE INTO reactions (guild_id, message_id, user_id, emoji, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (guild_id, message_id, user_id, emoji, now()),
    )
    _recount_reactions(guild_id, message_id)


def remove_reaction(guild_id: int, message_id: int, user_id: int, emoji: str) -> None:
    connect().execute(
        "DELETE FROM reactions WHERE guild_id=? AND message_id=? AND user_id=? AND emoji=?",
        (guild_id, message_id, user_id, emoji),
    )
    _recount_reactions(guild_id, message_id)


def clear_reactions(guild_id: int, message_id: int, emoji: Optional[str] = None) -> None:
    """Wipe every reaction on a message, or every use of one emoji on it."""
    if emoji is None:
        connect().execute(
            "DELETE FROM reactions WHERE guild_id=? AND message_id=?", (guild_id, message_id)
        )
    else:
        connect().execute(
            "DELETE FROM reactions WHERE guild_id=? AND message_id=? AND emoji=?",
            (guild_id, message_id, emoji),
        )
    _recount_reactions(guild_id, message_id)


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


def record_delivery(
    guild_id: int, kind: str, chars: int, journal_id: Optional[int] = None
) -> None:
    """Note that cadybot said something. Written by notify.deliver, nowhere else.

    Putting the write inside `deliver` rather than in each caller is the same
    move `notify._guard` makes: a new mouth is covered because it has to go
    through the one door, not because whoever added it remembered to log.
    """
    connect().execute(
        "INSERT INTO deliveries (guild_id, at, kind, chars, journal_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (guild_id, now(), kind, chars, journal_id),
    )


def deliveries_since(guild_id: int, since: str, kind: Optional[str] = None) -> int:
    """How many times cadybot has spoken since `since`, optionally of one kind."""
    if kind is None:
        return scalar(
            "SELECT COUNT(*) FROM deliveries WHERE guild_id=? AND at>=?",
            (guild_id, since),
        )
    return scalar(
        "SELECT COUNT(*) FROM deliveries WHERE guild_id=? AND at>=? AND kind=?",
        (guild_id, since, kind),
    )


def purge_member(guild_id: int, user_id: int) -> int:
    """Delete everything about one member. Data-deletion requests land here.

    Includes the journal, which holds redacted quotes of their messages and
    their display name inside a `finding`. A forget-me request that leaves a
    second copy of what somebody said in a different table has not been honoured
    — and the desk only started keeping that copy recently.
    """
    conn = connect()
    removed = 0
    for sql, params in (
        ("DELETE FROM messages WHERE guild_id=? AND author_id=?", (guild_id, user_id)),
        (
            "DELETE FROM mentions WHERE guild_id=? AND (author_id=? OR mentioned_id=?)",
            (guild_id, user_id, user_id),
        ),
        ("DELETE FROM reactions WHERE guild_id=? AND user_id=?", (guild_id, user_id)),
        ("DELETE FROM member_events WHERE guild_id=? AND user_id=?", (guild_id, user_id)),
        ("DELETE FROM voice_sessions WHERE guild_id=? AND user_id=?", (guild_id, user_id)),
        ("DELETE FROM members WHERE guild_id=? AND user_id=?", (guild_id, user_id)),
        # The desk stores redacted quotes of member messages and a display name
        # inside `finding`. Keyed by provocation rather than by user, so there is
        # nothing to match on — the honest move is to drop any journal row that
        # quoted anybody, since a forget-me request that leaves a second copy of
        # what somebody said in another table has not been honoured. The thought
        # is recoverable; the person's data is not theirs to keep.
        ("DELETE FROM journal WHERE guild_id=? AND ("
         "  finding IS NOT NULL OR to_founder IS NOT NULL) AND ? IS NOT NULL",
         (guild_id, user_id)),
    ):
        removed += conn.execute(sql, params).rowcount
    return removed


# Every table keyed by guild_id. Kept as a list rather than discovered at
# runtime so adding a table without deciding what leaving a server means to it
# is a visible omission instead of a silent leak.
GUILD_TABLES = (
    "messages", "mentions", "reactions", "member_events", "voice_sessions",
    "invite_uses", "recommendations", "runs", "conversation", "channels",
    "members", "guild_facts", "presence_samples", "audit_events", "deliveries",
    "ledger", "journal", "settings", "guilds",
)


def purge_guild(guild_id: int) -> int:
    """Forget one server completely. Being removed means being forgotten."""
    conn = connect()
    removed = 0
    for table in GUILD_TABLES:
        removed += conn.execute(
            "DELETE FROM %s WHERE guild_id=?" % table, (guild_id,)
        ).rowcount
    return removed


# The writable columns of guild_facts. Callers name a subset and the rest keep
# whatever they held, so a permission failure on one lookup cannot blank a fact
# a different lookup already established.
GUILD_FACT_COLUMNS = (
    "onboarding_enabled", "onboarding_mode", "onboarding_prompts",
    "onboarding_required_prompts", "onboarding_default_channels",
    "widget_enabled", "boost_count", "boost_tier", "verification_level",
    "is_community", "audit_readable", "onboarding_readable",
)


def upsert_guild_facts(guild_id: int, facts: Dict[str, Any]) -> None:
    cols = [c for c in GUILD_FACT_COLUMNS if c in facts]
    connect().execute(
        "INSERT INTO guild_facts (guild_id, at%s) VALUES (?, ?%s) "
        "ON CONFLICT(guild_id) DO UPDATE SET at=excluded.at%s"
        % (
            "".join(", " + c for c in cols),
            ", ?" * len(cols),
            "".join(", %s=excluded.%s" % (c, c) for c in cols),
        ),
        tuple([guild_id, now()] + [facts[c] for c in cols]),
    )


def record_presence_sample(
    guild_id: int, approx_members: Optional[int], approx_online: Optional[int]
) -> None:
    connect().execute(
        "INSERT INTO presence_samples (guild_id, at, approx_members, approx_online) "
        "VALUES (?, ?, ?, ?)",
        (guild_id, now(), approx_members, approx_online),
    )


def record_audit_event(
    guild_id: int,
    entry_id: int,
    action: int,
    user_id: Optional[int],
    target_id: Optional[int],
    at: Optional[str],
    extra_json: Optional[str] = None,
) -> None:
    connect().execute(
        "INSERT OR IGNORE INTO audit_events "
        "(guild_id, entry_id, action, user_id, target_id, at, extra_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (guild_id, entry_id, action, user_id, target_id, at or now(), extra_json),
    )


def last_audit_entry_id(guild_id: int, action: int) -> int:
    """Where a bounded audit catch-up should resume from. 0 means never seen."""
    return scalar(
        "SELECT MAX(entry_id) FROM audit_events WHERE guild_id=? AND action=?",
        (guild_id, action),
        default=0,
    )
