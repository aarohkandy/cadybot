"""The lookup tools. Deterministic, no model, no Discord, no network.

What is under test here is mostly structure rather than behaviour, because the
risky part of giving a model tools is not what the tools return — it is what a
tool could be talked into doing. A model chooses tool *arguments*, so the
guarantees have to hold for arguments the model invents.

Run:
    .venv/bin/python tests/prober.py
"""

import ast
import json
import os
import pathlib
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "prober.db"
os.environ["CADYBOT_DB"] = "prober.db"
os.environ.pop("GUILD_ID", None)
os.environ.pop("OWNER_ID", None)

from cadybot import db, probe  # noqa: E402

PASSED, FAILED = [], []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(("ok    " if cond else "FAIL  ") + name + ("" if cond else "   %s" % (detail,)))


def imports_of(path):
    tree = ast.parse(pathlib.Path(path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.name)
            if node.module:
                names.add(node.module.split(".")[0])
    return names


# --- structure: what a tool could be talked into ---------------------------

FORBIDDEN = {"discord", "notify", "room", "scorecard", "advisor", "agenda", "thinking"}
for mod in ("probe", "inquiry"):
    bad = imports_of(ROOT / "cadybot" / ("%s.py" % mod)) & FORBIDDEN
    check("1  %s.py cannot reach anything that writes or grades" % mod, not bad, sorted(bad))

check("2  scorecard stays independent of the tools",
      not (imports_of(ROOT / "cadybot" / "scorecard.py") & {"probe", "inquiry"}))

check("3  every tool is implemented in probe.py itself",
      all(t.fn.__module__ == "cadybot.probe" for t in probe.TOOLS.values()),
      [(n, t.fn.__module__) for n, t in probe.TOOLS.items()])

# The whole guild-isolation argument: the model fills in arguments, so guild_id
# must not be an argument. It is bound positionally by the dispatcher.
params = [(n, p.name) for n, t in probe.TOOLS.items() for p in t.params]
check("4  no tool takes a guild parameter",
      not [x for x in params if "guild" in x[1].lower()], params)
check("4b and no generated schema mentions one",
      not re.search(r"guild", json.dumps(probe.schemas()), re.I))

sql_blobs = pathlib.Path(ROOT / "cadybot" / "probe.py").read_text()
check("5  no server id is hardcoded anywhere in the SQL",
      not re.search(r"\b\d{16,}\b", sql_blobs))
check("6  the surface stays small", len(probe.TOOLS) <= 8, len(probe.TOOLS))

# --- the read-only driver ---------------------------------------------------

db.connect()
G, OTHER = 4242, 9999
conn = db.connect()
now = datetime.now(timezone.utc)
for gid, name in ((G, "ours"), (OTHER, "theirs")):
    conn.execute("INSERT OR REPLACE INTO guilds (guild_id, name, first_seen) VALUES (?,?,?)",
                 (gid, name, (now - timedelta(days=30)).isoformat()))
    conn.execute("INSERT OR REPLACE INTO channels (guild_id, channel_id, name, kind) "
                 "VALUES (?,?,?,'TextChannel')", (gid, gid + 1, "chan-" + name))
    conn.execute("INSERT OR REPLACE INTO members (guild_id, user_id, username, display_name,"
                 " is_bot, first_seen) VALUES (?,?,?,?,0,?)",
                 (gid, gid + 7, "u-" + name, "u-" + name, now.isoformat()))
    for i in range(3):
        conn.execute("INSERT OR REPLACE INTO messages (guild_id, channel_id, message_id,"
                     " author_id, created_at, content, type) VALUES (?,?,?,?,?,?,0)",
                     (gid, gid + 1, gid * 100 + i, gid + 7,
                      (now - timedelta(days=i)).isoformat(), "secret-of-" + name))

f = probe.run(G, "roster_authors", {})
check("7  a lookup sees only its own server",
      all("theirs" not in a["author"] for a in f.facts["authors"]), f.facts["authors"])

f = probe.run(G, "messages_search", {"term": "secret"})
check("7b and search cannot cross the boundary either",
      f.rows == 3 and all("theirs" not in q for q in f.quotes), (f.rows, f.quotes))

# An invented argument must not reach SQL.
f = probe.run(G, "messages_search", {"term": "secret", "guild_id": OTHER, "evil": "x"})
check("8  an invented guild_id argument is dropped, not honoured",
      f.rows == 3 and "guild_id" not in f.args, (f.rows, f.args))

check("9  an unknown tool name is a finding, not a crash",
      probe.run(G, "drop_everything", {}).error is not None)
check("9b a missing required argument too",
      probe.run(G, "channel_messages", {}).error is not None)
check("9c and a non-numeric one",
      probe.run(G, "channel_messages", {"ref": "the first one"}).error is not None)
check("9d out-of-range refs name the range, not the roster",
      "between 1 and" in (probe.run(G, "channel_messages", {"ref": 999}).error or ""))

# ints are clamped rather than trusted
f = probe.run(G, "messages_search", {"term": "secret", "limit": 10 ** 9})
check("10 an absurd limit is clamped", f.rows <= probe.PROBE_MAX_ROWS, f.rows)

# --- read-only is enforced at the driver, not by inspecting the SQL --------

ro = probe._ro()
for statement in ("DELETE FROM messages",
                  "UPDATE messages SET content='x'",
                  "INSERT INTO messages (guild_id, message_id, channel_id, author_id,"
                  " created_at) VALUES (1,2,3,4,'x')",
                  "PRAGMA writable_schema=1",
                  "PRAGMA journal_mode=DELETE"):
    try:
        ro.execute(statement)
        ok = False
    except sqlite3.DatabaseError:
        ok = True
    check("11 refused: %s" % statement.split()[0].lower(), ok)

check("11c but column introspection is allowed, since open_bets needs it",
      len(ro.execute("PRAGMA table_info(recommendations)").fetchall()) > 0)
check("11b but a plain read still works",
      ro.execute("SELECT COUNT(*) FROM messages WHERE guild_id=?", (G,)).fetchone()[0] == 3)

# --- facts vs quotes: the number rule --------------------------------------

f = probe.run(G, "open_bets", {})
check("12 open_bets survives a database scorecard has never touched",
      f.error is None and "recommendations" in f.facts, f.error)

from cadybot import advisor  # noqa: E402

conn.execute("INSERT INTO recommendations (guild_id, created_at, headline, action, metric) "
             "VALUES (?,?,?,?,?)",
             (G, now.isoformat(), "h", "Spend 90 minutes on r/3Dprinting", "none"))
f = probe.run(G, "open_bets", {})
known = set()
for block in [f.facts]:
    known |= advisor._numeric_literals(block)
check("13 a figure from stored advice stays uncitable",
      "90" not in known,
      "90 leaked from quotes into the citable set")
check("13b while a figure the SQL computed is citable",
      advisor._numeric_literals(probe.run(G, "roster_authors", {}).facts) & {"3"})

# --- empty is a result, not a silence ---------------------------------------

f = probe.run(G, "messages_search", {"term": "zzzznotpresent"})
check("14 zero rows is stated, not implied",
      f.rows == 0 and f.error is None and "confirmed empty" in f.body, f.body[:60])

# --- the loop, with no model available --------------------------------------

from cadybot import config, inquiry  # noqa: E402

_backend = config.BACKEND
config.BACKEND = "anthropic"
inq = inquiry.investigate(G, "anything", 2, 5)
check("15 the loop is a no-op on a backend without tools",
      inq.stopped == "off" and not inq.findings and inq.digest == "")
config.BACKEND = _backend

check("16 a footer is only written when something was checked",
      inquiry.footer([]) == "" and "checked" in inquiry.footer([probe.run(G, "open_bets", {})]))

check("17 a tool call written as prose is still recovered",
      inquiry._sniff('sure: {"name": "roster_authors", "arguments": {"days": 30}}')[0]
      ["function"]["name"] == "roster_authors")
check("17b and ordinary prose is not mistaken for one",
      inquiry._sniff("I do not need a lookup for that.") == [])

# --- redaction --------------------------------------------------------------

masked, n = probe._redact("mail me at a@b.com or https://x.co and 12345678")
check("18 contact details are masked before a model sees them",
      "a@b.com" not in masked and "https://x.co" not in masked and "12345678" not in masked
      and n == 3, masked)

print()
print("%d passed, %d failed" % (len(PASSED), len(FAILED)))
for n in FAILED:
    print("  failed:", n)
for suffix in ("", "-wal", "-shm"):
    p = pathlib.Path(str(DB) + suffix)
    if p.exists():
        p.unlink()
sys.exit(1 if FAILED else 0)
