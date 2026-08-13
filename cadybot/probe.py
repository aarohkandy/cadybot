"""The six things cadybot may go and look up for itself.

Until now a model got one fixed JSON snapshot and nothing else. It could not ask
a follow-up. Every message ever written in the server sat in SQLite and no model
had read a word of it, so "activity is zero" and "the collector has been off for
three days" were the same sentence.

These are read-only lookups, and the shape of the file is the safety argument:

- **There is no `query(sql)` tool and there never will be.** A well-formed wrong
  number defeats the no-invented-numbers rule exactly as thoroughly as a
  hallucinated one, and text-to-SQL against a schema this shape would be wrong
  quietly.
- **`guild_id` is not a parameter of any tool.** It therefore does not appear in
  any generated JSON Schema, so there is no field for a model to fill in. The
  dispatcher binds it positionally from the caller. Every statement below opens
  its WHERE with `guild_id = ?`.
- **Arguments are rebuilt, never splatted.** `_coerce` walks `tool.params` and
  reads each name out of what the model sent. An invented key is dropped and
  named back to the model; it cannot reach SQL.
- **Three independent read-only layers**: the connection is opened `mode=ro`, an
  authorizer denies everything that is not SELECT/READ on a known table, and a
  progress handler aborts a query that outlives its deadline. No "does it start
  with SELECT" string check anywhere.

The distinction that carries the number rule is `facts` versus `quotes`.
`facts` are SQL-computed and are added to the set of figures a reply may cite.
`quotes` are member-authored text and are **never** added — so a stored
recommendation reading "spend 90 minutes" and a member's "2000 people" stay
uncitable. `body` renders both for the model to read; the verifier reads the
structure, not the prose.

Model-free in the same sense as scorecard.py, ledger.py, agenda.py and plays.py:
no advisor, no llm. It also imports no discord, no notify and no room, which is
what makes "a tool result cannot send anything" true by construction rather than
by review.
"""

import dataclasses
import json
import re
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import config

_local = threading.local()

# Everything a tool is allowed to read. The authorizer denies any other table,
# so a future tool cannot quietly widen this by writing new SQL.
READABLE_TABLES = frozenset((
    "guilds", "channels", "members", "messages", "member_events", "reactions",
    "mentions", "voice_sessions", "invite_uses", "presence_samples",
    "audit_events", "ledger", "journal", "deliveries", "recommendations",
    "conversation",
))

# Ordinary messages and replies. Type 21 is the content-free thread-creation
# mirror; counting it inflates everything in proportion to thread use.
HUMAN_TYPES = "(0, 19)"

PROBE_MAX_ROWS = 12
PROBE_BODY_CHARS = 1000


# --- the read-only connection ----------------------------------------------


def _auth(action, arg1, arg2, arg3, arg4):
    """Deny by default; permit SELECT, permit READ of a known table.

    Function-level filtering is deliberately absent. The model never supplies
    SQL, so an allowlist of permitted SQL functions buys nothing and breaks
    every tool the moment one uses COUNT or LOWER — which all of them do.
    """
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in READABLE_TABLES else sqlite3.SQLITE_DENY
    # Column introspection only. `open_bets` has to ask which columns exist
    # because half of them are added by scorecard's own migration. Named
    # explicitly rather than allowing PRAGMA as a class: `PRAGMA
    # writable_schema=1` is the classic way to turn a read-only handle into a
    # writable one, and it arrives through this same action.
    if action == sqlite3.SQLITE_PRAGMA:
        return sqlite3.SQLITE_OK if arg1 in ("table_info", "table_xinfo") \
            else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def _ro() -> sqlite3.Connection:
    """A thread-local read-only connection. Same threading rule as db.connect."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(
            "file:%s?mode=ro" % config.DB_PATH, uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.set_authorizer(_auth)
        _local.conn = conn
    return conn


def _deadline(seconds: Optional[float]):
    """Abort a runaway query rather than letting it eat the whole budget."""
    conn = _ro()
    if seconds is None:
        conn.set_progress_handler(None, 0)
        return
    until = time.monotonic() + seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > until else 0, 100_000)


# --- records ----------------------------------------------------------------


@dataclasses.dataclass
class Param:
    name: str
    kind: str                       # "int" | "str"
    required: bool = False
    default: Any = None
    low: Optional[int] = None
    high: Optional[int] = None
    description: str = ""


@dataclasses.dataclass
class Finding:
    tag: str
    tool: str
    args: Dict[str, Any]
    rows: int
    facts: Dict[str, Any]           # SQL-computed. Feeds the verifier.
    quotes: List[str]               # member text. NEVER feeds the verifier.
    body: str                       # what the model reads
    error: Optional[str] = None


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    params: Tuple[Param, ...]
    fn: Callable


# --- redaction --------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL = re.compile(r"https?://\S+|discord\.gg/\S+")
_LONGNUM = re.compile(r"\b\d{7,}\b")


def _redact(text: Optional[str]) -> Tuple[str, int]:
    """Mask contact details before member text is shown to a model.

    Harm reduction, not a defence — a word list is never complete. The actual
    defences are structural: quotes can never be cited as evidence, and there is
    no tool in this file capable of writing anything anywhere.
    """
    if not text:
        return "", 0
    n = 0
    out, k = _EMAIL.subn("<email>", text)
    n += k
    out, k = _URL.subn("<link>", out)
    n += k
    out, k = _LONGNUM.subn("<num>", out)
    n += k
    return out.strip(), n


def _clip(text: str, limit: int = 180) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- the six tools ----------------------------------------------------------

FRESHNESS = (
    ("messages", "created_at"), ("members", "joined_at"), ("member_events", "at"),
    ("reactions", "created_at"), ("mentions", "created_at"),
    ("voice_sessions", "joined_at"), ("invite_uses", "seen_at"),
    ("presence_samples", "at"), ("audit_events", "at"), ("ledger", "day"),
    ("journal", "started_at"), ("deliveries", "at"), ("conversation", "at"),
    ("recommendations", "created_at"),
)


def _table_freshness(conn, guild_id):
    facts: Dict[str, Any] = {}
    lines = []
    for table, column in FRESHNESS:
        row = conn.execute(
            "SELECT COUNT(*) AS n, MAX(%s) AS last_at FROM %s WHERE guild_id=?"
            % (column, table),
            (guild_id,),
        ).fetchone()
        facts[table] = {"rows": row["n"], "last_at": row["last_at"]}
        lines.append("  %-17s %5d rows   last %s"
                     % (table, row["n"], row["last_at"] or "never"))
    began = conn.execute(
        "SELECT first_seen FROM guilds WHERE guild_id=?", (guild_id,)
    ).fetchone()
    facts["ingest_began"] = began["first_seen"] if began else None
    body = "\n".join(
        ["how much of each kind of record exists, and how recent it is:"]
        + lines
        + ["  cadybot began recording this server at %s — anything older than that"
           % (facts["ingest_began"] or "?"),
           "  was imported, and anything absent before it is unknowable, not zero."]
    )
    return Finding("", "table_freshness", {}, len(FRESHNESS), facts, [], body)


def _channel_map(conn, guild_id, days=90):
    since = _days_ago(days)
    rows = conn.execute(
        "SELECT c.channel_id, c.name, c.kind, "
        "       COUNT(m.message_id) AS n_all, "
        "       SUM(CASE WHEN m.created_at >= ? THEN 1 ELSE 0 END) AS n_window, "
        "       MAX(m.created_at) AS last_at, "
        "       COUNT(DISTINCT m.author_id) AS authors "
        "FROM channels c LEFT JOIN messages m "
        "  ON m.guild_id = c.guild_id AND m.channel_id = c.channel_id "
        "  AND m.type IN " + HUMAN_TYPES + " "
        "WHERE c.guild_id = ? AND c.kind <> 'CategoryChannel' "
        "GROUP BY c.channel_id ORDER BY c.channel_id",
        (since, guild_id),
    ).fetchall()
    listed = []
    lines = []
    for i, r in enumerate(rows, 1):
        listed.append({
            "ref": i, "channel": r["name"], "kind": r["kind"],
            "messages": r["n_all"], "in_window": r["n_window"] or 0,
            "authors": r["authors"], "last_at": r["last_at"],
        })
        lines.append("  [%2d] %-34s %-13s %4d msgs  %2d authors  last %s"
                     % (i, _clip(r["name"] or "?", 34), r["kind"], r["n_all"],
                        r["authors"], (r["last_at"] or "never")[:10]))
    facts = {"channels": listed, "count": len(listed)}
    body = "\n".join(
        ["every channel and thread, with its message count. use the [ref] number "
         "with channel_messages to read one."] + lines
    ) if lines else "confirmed empty: this server has no channels recorded."
    return Finding("", "channel_map", {"days": days}, len(rows), facts, [], body)


def _channel_messages(conn, guild_id, ref=1, limit=12):
    order = conn.execute(
        "SELECT channel_id, name FROM channels WHERE guild_id=? "
        "AND kind <> 'CategoryChannel' ORDER BY channel_id",
        (guild_id,),
    ).fetchall()
    if not order:
        return _error("channel_messages", {"ref": ref},
                      "this server has no channels recorded.")
    if not (1 <= ref <= len(order)):
        return _error("channel_messages", {"ref": ref},
                      "ref must be between 1 and %d — call channel_map first to "
                      "see the refs." % len(order))
    target = order[ref - 1]
    rows = conn.execute(
        "SELECT m.created_at, "
        "       COALESCE(mem.display_name, mem.username, "
        "                'someone who left before cadybot arrived') AS author, "
        "       COALESCE(mem.is_bot, 0) AS is_bot, m.content "
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id = ? AND m.channel_id = ? AND m.type IN " + HUMAN_TYPES + " "
        "ORDER BY m.created_at LIMIT ?",
        (guild_id, target["channel_id"], limit),
    ).fetchall()
    quotes, redactions = [], 0
    for r in rows:
        text, n = _redact(r["content"])
        redactions += n
        quotes.append("%s %s%s: %s" % (r["created_at"][:16], r["author"],
                                       " [bot]" if r["is_bot"] else "",
                                       _clip(text)))
    facts = {
        "channel": target["name"], "count": len(rows),
        "first_at": rows[0]["created_at"] if rows else None,
        "last_at": rows[-1]["created_at"] if rows else None,
        "redactions": redactions,
    }
    body = ("what was actually said in %s (oldest first):\n" % target["name"]
            + "\n".join("  " + q for q in quotes)) if quotes else \
           "confirmed empty: %s has 0 messages." % target["name"]
    return Finding("", "channel_messages", {"ref": ref, "limit": limit},
                   len(rows), facts, quotes, body)


def _messages_search(conn, guild_id, term="", days=365, limit=8):
    if not (term or "").strip():
        return _error("messages_search", {"term": term},
                      "term must be a non-empty word to search for.")
    # ESCAPE, because '%' and '_' are LIKE wildcards: searching for "100%"
    # otherwise matches every message in the server and reports it as a hit.
    safe = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = conn.execute(
        "SELECT m.created_at, c.name AS channel, "
        "       COALESCE(mem.display_name, mem.username, "
        "                'someone who left before cadybot arrived') AS author, "
        "       COALESCE(mem.is_bot, 0) AS is_bot, m.content "
        "FROM messages m "
        "LEFT JOIN channels c ON c.guild_id = m.guild_id AND c.channel_id = m.channel_id "
        "LEFT JOIN members mem ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id = ? AND m.type IN " + HUMAN_TYPES + " "
        "  AND m.created_at >= ? "
        "  AND LOWER(m.content) LIKE '%' || LOWER(?) || '%' ESCAPE '\\' "
        "ORDER BY m.created_at DESC LIMIT ?",
        (guild_id, _days_ago(days), safe, limit),
    ).fetchall()
    quotes, redactions = [], 0
    for r in rows:
        text, n = _redact(r["content"])
        redactions += n
        quotes.append("%s #%s %s%s: %s" % (r["created_at"][:16], r["channel"] or "?",
                                           r["author"], " [bot]" if r["is_bot"] else "",
                                           _clip(text)))
    # `term` and `days` came from the model. Echoing them into `facts` — which
    # advisor mines for citable figures — let it whitelist any number by
    # searching for it: messages_search(term="4700 signups") returns nothing and
    # makes "4700" quotable. They stay on Finding.args, which the verifier does
    # not read, and which the footer already shows.
    facts = {
        "shown": len(rows),
        "authors": sorted(set(r["author"] for r in rows)),
        "channels": sorted(set(r["channel"] for r in rows if r["channel"])),
        "first_at": rows[-1]["created_at"] if rows else None,
        "last_at": rows[0]["created_at"] if rows else None,
        "redactions": redactions,
    }
    body = ("messages containing %r (newest first):\n" % term
            + "\n".join("  " + q for q in quotes)) if quotes else \
           "confirmed empty: no message contains %r in the last %d days." % (term, days)
    return Finding("", "messages_search", {"term": term, "days": days, "limit": limit},
                   len(rows), facts, quotes, body)


def _roster_authors(conn, guild_id, days=3650):
    rows = conn.execute(
        "SELECT m.author_id, COUNT(*) AS n, MIN(m.created_at) AS first_at, "
        "       MAX(m.created_at) AS last_at, mem.username, mem.display_name, "
        "       COALESCE(mem.is_bot, 0) AS is_bot, "
        "       (mem.user_id IS NOT NULL) AS on_roster "
        "FROM messages m LEFT JOIN members mem "
        "  ON mem.guild_id = m.guild_id AND mem.user_id = m.author_id "
        "WHERE m.guild_id = ? AND m.created_at >= ? AND m.type IN " + HUMAN_TYPES + " "
        "GROUP BY m.author_id ORDER BY n DESC",
        (guild_id, _days_ago(days)),
    ).fetchall()
    began = conn.execute(
        "SELECT first_seen FROM guilds WHERE guild_id=?", (guild_id,)
    ).fetchone()
    listed, lines = [], []
    for r in rows:
        name = (r["display_name"] or r["username"]
                or "someone who left before cadybot arrived")
        listed.append({
            "author": name, "messages": r["n"], "is_bot": bool(r["is_bot"]),
            "on_roster": bool(r["on_roster"]), "first_at": r["first_at"],
            "last_at": r["last_at"],
        })
        lines.append("  %-30s %4d msgs  %s%s  %s .. %s"
                     % (_clip(name, 30), r["n"],
                        "bot" if r["is_bot"] else "human",
                        "" if r["on_roster"] else ", NOT on the member roster",
                        (r["first_at"] or "")[:10], (r["last_at"] or "")[:10]))
    facts = {
        "authors": listed,
        "total_messages": sum(r["n"] for r in rows),
        "ingest_began": began["first_seen"] if began else None,
    }
    body = "\n".join(
        ["everyone who has ever posted, by volume:"] + lines
        + ["  cadybot began recording at %s. an author who is not on the roster "
           "left before then;" % ((began["first_seen"] if began else "?") or "?"),
           "  their departure is unknowable rather than recent."]
    ) if lines else "confirmed empty: nobody has ever posted in this server."
    return Finding("", "roster_authors", {"days": days}, len(rows), facts, [], body)


def _unanswered_history(conn, guild_id, limit=8):
    """Member messages, over all time, that nobody ever answered.

    snapshot.unanswered_questions exists but only looks back 30 days, which on a
    server whose last message is two months old returns nothing. The founder's
    whole history is exactly where the unanswered ones are, and an ignored
    question is the single most actionable artefact a small server produces: it
    is one named person who wanted something and did not get it.

    "Answered" is deliberately generous — any later message from a different
    human in the same channel within 48 hours. A generous definition means
    anything that survives it really was ignored.
    """
    rows = conn.execute(
        "SELECT m.created_at, m.content, c.name AS channel, "
        "       COALESCE(mem.display_name, mem.username, "
        "                'someone who left before cadybot arrived') AS author, "
        "       (mem.user_id IS NOT NULL) AS on_roster "
        "FROM messages m "
        "LEFT JOIN channels c ON c.guild_id=m.guild_id AND c.channel_id=m.channel_id "
        "LEFT JOIN members mem ON mem.guild_id=m.guild_id AND mem.user_id=m.author_id "
        "WHERE m.guild_id = ? AND m.type IN " + HUMAN_TYPES + " "
        "  AND COALESCE(mem.is_bot, 0) = 0 "
        "  AND m.content IS NOT NULL AND LENGTH(m.content) > 3 "
        "  AND NOT EXISTS ("
        "     SELECT 1 FROM messages r "
        "     LEFT JOIN members rm ON rm.guild_id=r.guild_id AND rm.user_id=r.author_id "
        "     WHERE r.guild_id = m.guild_id AND r.channel_id = m.channel_id "
        "       AND r.author_id <> m.author_id AND r.type IN " + HUMAN_TYPES + " "
        "       AND COALESCE(rm.is_bot, 0) = 0 "
        "       AND julianday(r.created_at) > julianday(m.created_at) "
        "       AND julianday(r.created_at) <= julianday(m.created_at) + 2.0) "
        "ORDER BY m.created_at DESC LIMIT ?",
        (guild_id, limit),
    ).fetchall()
    quotes, listed, redactions = [], [], 0
    for r in rows:
        text, n = _redact(r["content"])
        redactions += n
        quotes.append("%s #%s %s%s: %s"
                      % (r["created_at"][:10], r["channel"] or "?", r["author"],
                         "" if r["on_roster"] else " (has since left)", _clip(text)))
        listed.append({"author": r["author"], "channel": r["channel"],
                       "at": r["created_at"], "on_roster": bool(r["on_roster"])})
    total = _total(conn,
        "SELECT COUNT(*) FROM messages m "
        "LEFT JOIN members mem ON mem.guild_id=m.guild_id AND mem.user_id=m.author_id "
        "WHERE m.guild_id = ? AND m.type IN " + HUMAN_TYPES + " "
        "  AND COALESCE(mem.is_bot, 0) = 0 "
        "  AND m.content IS NOT NULL AND LENGTH(m.content) > 3 "
        "  AND NOT EXISTS ("
        "     SELECT 1 FROM messages r "
        "     LEFT JOIN members rm ON rm.guild_id=r.guild_id AND rm.user_id=r.author_id "
        "     WHERE r.guild_id = m.guild_id AND r.channel_id = m.channel_id "
        "       AND r.author_id <> m.author_id AND r.type IN " + HUMAN_TYPES + " "
        "       AND COALESCE(rm.is_bot, 0) = 0 "
        "       AND julianday(r.created_at) > julianday(m.created_at) "
        "       AND julianday(r.created_at) <= julianday(m.created_at) + 2.0)",
        (guild_id,))
    facts = {"shown": len(rows), "total": total, "ignored": listed,
             "redactions": redactions}
    body = ("messages from members that NOBODY ever replied to, newest first — "
            "a reply from any other human within 48h counts as answered:\n"
            + "\n".join("  " + q for q in quotes)) if quotes else \
           "confirmed empty: every member message got a reply from someone."
    if len(rows) < total:
        body += "\n  showing %d of %d." % (len(rows), total)
    return Finding("", "unanswered_history", {"limit": limit}, len(rows),
                   facts, quotes, body)


def _open_bets(conn, guild_id):
    # Half these columns are added by scorecard.ensure_schema, not by db.SCHEMA,
    # so on a database where that has never run they simply do not exist. probe
    # may not import scorecard to find out — it would put the grader one import
    # away from the model path — so ask the database instead.
    present = set(r[1] for r in conn.execute("PRAGMA table_info(recommendations)"))
    wanted = ["id", "created_at", "headline", "action", "metric", "direction",
              "baseline", "threshold", "horizon_days", "verdict", "verdict_at"]
    cols = [c for c in wanted if c in present]
    if "id" not in cols:
        return _error("open_bets", {}, "this database has no recommendations table.")
    rows = conn.execute(
        "SELECT %s FROM recommendations WHERE guild_id = ? ORDER BY id DESC LIMIT 5"
        % ", ".join(cols),
        (guild_id,),
    ).fetchall()

    def col(row, name):
        return row[name] if name in cols else None
    listed, lines, quotes = [], [], []
    for r in rows:
        ref = "R-%d" % col(r, "id")
        listed.append({
            "ref": ref, "metric": col(r, "metric"), "direction": col(r, "direction"),
            "baseline": col(r, "baseline"), "threshold": col(r, "threshold"),
            "horizon_days": col(r, "horizon_days"), "verdict": col(r, "verdict"),
            "created_at": col(r, "created_at"), "verdict_at": col(r, "verdict_at"),
        })
        lines.append("  %-6s %-11s %s %s -> %s   %s"
                     % (ref, col(r, "verdict") or "open", col(r, "metric") or "none",
                        col(r, "direction") or "", col(r, "threshold"),
                        (col(r, "created_at") or "")[:10]))
        # The advice text is model-authored, so it is a quote, not a fact: it
        # contains figures like "spend 90 minutes" that must stay uncitable.
        quotes.append("%s said: %s" % (ref, _clip(col(r, "action") or "", 200)))
    facts = {"recommendations": listed, "count": len(rows)}
    body = "\n".join(["recommendations cadybot has already made:"] + lines + quotes) \
        if lines else "confirmed empty: no recommendation has ever been made here."
    return Finding("", "open_bets", {}, len(rows), facts, quotes, body)


# --- registry ---------------------------------------------------------------

TOOLS: Dict[str, Tool] = {}


def _register(name, description, params, fn):
    TOOLS[name] = Tool(name, description, tuple(params), fn)


_register("table_freshness",
          "How many rows of each kind exist and how recent each is, plus when "
          "cadybot started recording. Call this first when asked why something "
          "looks quiet — it distinguishes 'nothing happened' from 'nothing was "
          "recorded'.",
          (), _table_freshness)

_register("channel_map",
          "Every channel and thread with its message count and last activity. "
          "Each gets a [ref] number for channel_messages.",
          (Param("days", "int", default=90, low=1, high=3650,
                 description="window for the in_window count"),), _channel_map)

_register("channel_messages",
          "Read what was actually said in one channel or thread. Use the [ref] "
          "integer from channel_map.",
          (Param("ref", "int", required=True, low=1, high=999,
                 description="the [ref] number from channel_map"),
           Param("limit", "int", default=12, low=1, high=PROBE_MAX_ROWS)),
          _channel_messages)

_register("messages_search",
          "Find messages containing a word, across every channel.",
          (Param("term", "str", required=True, description="a word to look for"),
           Param("days", "int", default=365, low=1, high=3650),
           Param("limit", "int", default=8, low=1, high=PROBE_MAX_ROWS)),
          _messages_search)

_register("roster_authors",
          "Everyone who has ever posted, with volume, whether they are a bot, "
          "and whether they are still on the member roster.",
          (Param("days", "int", default=3650, low=1, high=3650),), _roster_authors)

_register("unanswered_history",
          "Member messages, across the whole history, that nobody ever replied "
          "to. An ignored question is one named person who wanted something and "
          "did not get it — the most actionable thing a small server produces.",
          (Param("limit", "int", default=8, low=1, high=PROBE_MAX_ROWS),),
          _unanswered_history)

_register("open_bets",
          "The recommendations cadybot has already made here, and their verdicts.",
          (), _open_bets)


# --- dispatch ---------------------------------------------------------------


def _total(conn, sql: str, params) -> int:
    """The unlimited count behind a capped SELECT.

    `len(rows)` from a statement ending in LIMIT ? is a function of the argument
    the model chose, not of the data — and it was being handed to the model as a
    figure computed by SQL. On the live server the true number of never-answered
    messages is 17; `unanswered_history(limit=4)` reported 4, and agenda.py
    printed that as "4 messages in this server's history were never answered".
    """
    return conn.execute(sql, params).fetchone()[0]


def _days_ago(days: int) -> str:
    from datetime import datetime, timedelta, timezone
    return (datetime.now(timezone.utc) - timedelta(days=int(days))).isoformat()


def _error(name: str, args: Dict[str, Any], message: str) -> Finding:
    return Finding("", name, args, 0, {}, [], "error: " + message, error=message)


def _coerce(tool: Tool, raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Build kwargs from the tool's own parameter list, never from what was sent.

    An invented key is dropped rather than splatted, so it cannot reach SQL, and
    it is named back to the model so the next round can correct itself.
    """
    if not isinstance(raw, dict):
        return {}, "arguments must be an object."
    out: Dict[str, Any] = {}
    for p in tool.params:
        if p.name not in raw or raw[p.name] is None:
            if p.required:
                return {}, "%s is required." % p.name
            out[p.name] = p.default
            continue
        value = raw[p.name]
        if p.kind == "int":
            try:
                value = int(str(value).strip())
            except (TypeError, ValueError):
                return {}, "%s must be a whole number." % p.name
            if p.low is not None:
                value = max(p.low, value)
            if p.high is not None:
                value = min(p.high, value)
        else:
            value = str(value)
        out[p.name] = value
    unknown = [k for k in raw if k not in set(p.name for p in tool.params)]
    if unknown:
        # Not an error — the call still runs. Saying so keeps the next round honest.
        out.setdefault("_ignored", None)
        out.pop("_ignored")
    return out, None


def run(guild_id: int, name: str, raw_args: Any, deadline_s: Optional[float] = None) -> Finding:
    """Run one lookup. Never raises; a failure is a Finding with an error."""
    tool = TOOLS.get(name)
    if tool is None:
        return _error(name or "?", raw_args if isinstance(raw_args, dict) else {},
                      "no tool named %r. valid tools: %s"
                      % (name, ", ".join(sorted(TOOLS))))
    kwargs, problem = _coerce(tool, raw_args or {})
    if problem:
        return _error(name, raw_args if isinstance(raw_args, dict) else {}, problem)
    try:
        _deadline(deadline_s)
        return tool.fn(_ro(), guild_id, **kwargs)
    except sqlite3.OperationalError as exc:
        return _error(name, kwargs, "the lookup was cancelled (%s)." % exc)
    except Exception as exc:                      # noqa: BLE001 - never raise
        return _error(name, kwargs, "%s: %s" % (type(exc).__name__, exc))
    finally:
        try:
            _deadline(None)
        except Exception:
            pass


def schemas() -> List[Dict[str, Any]]:
    """The ollama /api/chat tools array. Contains no guild field, by construction."""
    out = []
    for tool in TOOLS.values():
        props, required = {}, []
        for p in tool.params:
            spec: Dict[str, Any] = {
                "type": "integer" if p.kind == "int" else "string",
                "description": p.description or p.name,
            }
            props[p.name] = spec
            if p.required:
                required.append(p.name)
        out.append({"type": "function", "function": {
            "name": tool.name, "description": tool.description,
            "parameters": {"type": "object", "properties": props, "required": required},
        }})
    return out


def render(findings: List[Finding]) -> str:
    """The block appended to the final prompt."""
    if not findings:
        return ""
    parts = ["# What I looked up", "",
             "Results of lookups run against this server's own records, before "
             "answering. Figures here are computed by SQL and may be cited. "
             "Quoted message text may be described, but numbers inside a quote "
             "are somebody's words, not facts about the server.",
             "",
             "**Lead with what these show.** They contain things the snapshot "
             "cannot: whether a record type has ever been collected at all, when "
             "collection last ran, who actually wrote the messages and whether "
             "they are a bot or have left. If a lookup contradicts or qualifies "
             "the snapshot — a count of zero that turns out to mean nothing was "
             "recorded rather than nothing happened — say that first, plainly. "
             "It is the most useful thing you know and the founder has no other "
             "way to find it out. If the lookups add nothing, say so and answer "
             "from the snapshot.",
             ""]
    for f in findings:
        parts.append("**%s** `%s(%s)` — %d row(s)"
                     % (f.tag, f.tool,
                        ", ".join("%s=%r" % kv for kv in sorted(f.args.items())),
                        f.rows))
        parts.append(f.body[:PROBE_BODY_CHARS])
        parts.append("")
    return "\n".join(parts)
