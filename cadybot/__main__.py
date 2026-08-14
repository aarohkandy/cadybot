"""Command line entry point: python -m cadybot <command>

cadybot can be in several servers at once, so every command that reads data
needs to know which one. It resolves that from --guild, then GUILD_ID in .env,
then — if it only knows about one server — that one.
"""

import argparse
import asyncio
import json
import sys
import traceback
from typing import Optional

import discord

from . import (
    advisor, agenda, backfill, config, db, ledger, listener, llm, loop, room,
    scorecard, snapshot, thinking,
)


def _known_guilds():
    return db.query("SELECT guild_id, name FROM guilds ORDER BY name")


def _resolve_guild(explicit: Optional[int]) -> int:
    if explicit:
        return explicit
    if config.GUILD_ID:
        return config.GUILD_ID
    rows = _known_guilds()
    if len(rows) == 1:
        return rows[0]["guild_id"]
    if not rows:
        raise SystemExit(
            "cadybot hasn't seen any servers yet. Run `python -m cadybot listen` "
            "once after inviting it."
        )
    listing = "\n".join("  %s  %s" % (r["guild_id"], r["name"]) for r in rows)
    raise SystemExit("Several servers known — pass --guild <id>:\n%s" % listing)


def _one_shot_backfill(guild_id: int) -> None:
    """Connect, import that server's history, disconnect."""
    config.require_discord()
    client = discord.Client(intents=listener.INTENTS)

    @client.event
    async def on_ready():
        guild = client.get_guild(guild_id)
        if guild is None:
            print("cadybot is not in server %s." % guild_id)
        else:
            db.upsert_guild(guild.id, guild.name)
            total = await backfill.run(guild, skip_channel_id=room.stored_id(guild.id))
            print("Imported %d messages from %s." % (total, guild.name))
        await client.close()

    client.run(config.DISCORD_TOKEN, log_handler=None)


# View Channels | Send Messages | Read Message History | Manage Channels
# | Manage Roles | Manage Server. The two Manage permissions let cadybot create
# its private channel and control who sees it; Manage Server lets it read invite
# use-counts for join attribution.
SCOPED_PERMS = 1024 | 2048 | 65536 | 16 | 268435456 | 32
ADMIN_PERMS = 8


def _application_id() -> str:
    """The bot token's first segment is the application ID, base64-encoded."""
    import base64

    head = config.DISCORD_TOKEN.split(".")[0]
    padded = head + "=" * (-len(head) % 4)
    return base64.b64decode(padded).decode()


def _invite() -> int:
    if not config.DISCORD_TOKEN:
        print("Set DISCORD_TOKEN in .env first.")
        return 1
    try:
        app_id = _application_id()
    except Exception:
        print("Couldn't read the application ID from DISCORD_TOKEN. Is it a bot token?")
        return 1

    url = "https://discord.com/oauth2/authorize?client_id=%s&scope=bot+applications.commands&permissions=%d"
    print("Everything cadybot needs, and nothing more:")
    print("  " + url % (app_id, SCOPED_PERMS))
    print()
    print("Administrator (test servers only — grants far more than it uses):")
    print("  " + url % (app_id, ADMIN_PERMS))
    print()
    print("Add it to as many servers as you like; each one is kept separate.")
    print("Then run /private in each server to create your channel there.")
    return 0


def _post(kind: str, guild_id: int) -> int:
    """Run one scheduled pass and deliver it, then exit.

    The listener holds the long-lived gateway connection; this opens a second,
    short one just long enough to post. Discord allows concurrent sessions for a
    bot — that is how sharding works — so this does not disturb the listener.

    Delivery is the point. `cadybot loop` computes the same pass with client=None
    and deliberately does not deliver, which is what you want when testing.
    """
    config.require_discord()
    from . import loop as passes

    client = discord.Client(intents=listener.INTENTS)
    status = {"code": 1}

    @client.event
    async def on_ready():
        try:
            if client.get_guild(guild_id) is None:
                print("cadybot is not in server %s" % guild_id)
                return
            text = await (passes.weekly if kind == "weekly" else passes.nightly)(
                client, guild_id
            )
            print("delivered %d chars" % len(text) if text else "nothing worth saying")
            status["code"] = 0
        except Exception:
            traceback.print_exc()
        finally:
            await client.close()

    client.run(config.DISCORD_TOKEN, log_handler=None)
    return status["code"]


def _think(guild_id: int) -> int:
    """Run one pass of the desk and deliver anything it decides is worth saying.

    Same short second gateway session as `_post`, for the same reason. This is
    the cron entry point when CADYBOT_SCHEDULER=cron; under the default
    scheduler the listener runs it in-process and this is for running by hand.
    """
    config.require_discord()

    client = discord.Client(intents=listener.INTENTS)
    status = {"code": 1}

    @client.event
    async def on_ready():
        try:
            if client.get_guild(guild_id) is None:
                print("cadybot is not in server %s" % guild_id)
                return
            text = await thinking.think(client, guild_id)
            print("said %d chars" % len(text) if text else "thought nothing worth saying")
            status["code"] = 0
        except Exception:
            traceback.print_exc()
        finally:
            await client.close()

    client.run(config.DISCORD_TOKEN, log_handler=None)
    return status["code"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cadybot")
    sub = parser.add_subparsers(dest="command", required=True)

    def with_guild(name: str, help_text: str):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--guild", type=int, help="server id (see `servers`)")
        return p

    listen = sub.add_parser("listen", help="run the gateway listener (do this first)")
    listen.add_argument("--backfill", action="store_true", help="import history on startup")

    sub.add_parser("invite", help="print the OAuth invite URL")
    sub.add_parser("doctor", help="check the model backend is reachable")
    sub.add_parser("servers", help="list the servers cadybot knows about")

    with_guild("backfill", "import history and exit")
    with_guild("snapshot", "print the raw numbers, no LLM")

    led = with_guild("ledger", "the daily history of one metric, no LLM")
    led.add_argument("--metric", help="a ledger.LEDGER_METRICS path")
    led.add_argument("--days", type=int, default=30)

    with_guild("brief", "ranked recommendations")
    with_guild("outcomes", "past recommendations and their outcomes")
    with_guild("score", "grade recommendations past their horizon, no LLM")
    with_guild("loop", "run one nightly pass without delivering it")

    post = with_guild("post", "run a scheduled pass AND deliver it (for cron)")
    post.add_argument("kind", choices=["nightly", "weekly"])

    with_guild("reflect", "what the desk would think about right now, no LLM")
    with_guild("think", "run one pass of the desk AND deliver it (for cron)")
    with_guild("notes", "everything the desk thought, said or not")

    ask = with_guild("ask", "a straight yes / no / not-yet")
    ask.add_argument("question", nargs="+")

    purge = with_guild("purge", "delete one member's data")
    purge.add_argument("--member", type=int, required=True, help="Discord user id")

    args = parser.parse_args(argv)

    if args.command == "listen":
        listener.run(backfill_on_start=args.backfill)
        return 0

    if args.command == "invite":
        return _invite()

    if args.command == "doctor":
        print("backend: %s" % llm.describe())
        problem = llm.preflight()
        print(problem if problem else "backend reachable.")
        return 1 if problem else 0

    if args.command == "servers":
        rows = _known_guilds()
        if not rows:
            print("None yet. Invite cadybot, then run `python -m cadybot listen`.")
            return 0
        for r in rows:
            gid = r["guild_id"]
            has_room = room.stored_id(gid)
            owner = room.owner_id(gid)
            print(
                "%s  %-28s %s"
                % (
                    gid,
                    r["name"],
                    "private channel set up (owner %s)" % owner if has_room else "no /private yet",
                )
            )
        return 0

    guild_id = _resolve_guild(args.guild)

    if args.command == "post":
        return _post(args.kind, guild_id)

    if args.command == "backfill":
        _one_shot_backfill(guild_id)
        return 0

    if args.command == "snapshot":
        print(json.dumps(snapshot.build(guild_id), indent=2, default=str))
        return 0

    if args.command == "reflect":
        state = thinking.preview(guild_id)
        print("looked      : %d times today, last at %s"
              % (state.get("scans_today", 0), (state.get("last_look") or "-")[11:19]))
        print("budget      : %d of %d used in the last 24h%s"
              % (state["spent_today"], state["budget"],
                 "" if state["affordable"] else "  (spent)"))
        if state["provocation"] is None:
            print("provocation : none — %s" % state["why"])
            return 0
        print("provocation : %s, from %s%s"
              % (state["provocation"], state["provoked_by"],
                 " (%s)" % state["about"] if state["about"] else ""))
        print("would speak : %s — %s"
              % ("yes" if state["would_surface"] else "no", state["why"]))
        print("\n--- the question it would ask itself ---\n")
        print(state["self_prompt"])
        return 0

    if args.command == "think":
        return _think(guild_id)

    if args.command == "notes":
        rows = agenda.recent(guild_id, limit=20)
        if not rows:
            print("The desk has not thought about anything yet.")
            print("Run `reflect` to see what it is waiting for.")
            return 0
        for row in rows:
            mark = {"thought": " ", "failed": "!", "started": "~"}.get(row["outcome"], "?")
            said = "spoke" if row["surfaced_at"] else "kept it"
            print("%s %-8s %s  %s" % (mark, row["kind"], row["started_at"][:16], said))
            if row["about_ref"]:
                print("    about   %s" % row["about_ref"])
            if row["failure"]:
                print("    failed  %s" % row["failure"])
            if row["note_to_self"]:
                print("    note    %s" % row["note_to_self"])
            if row["to_founder"] and not row["surfaced_at"]:
                # Say why it was withheld. Printing the sentence bare made a
                # thought suppressed for citing an unverifiable figure look
                # identical to one merely waiting for a good moment.
                unver = (row["unverified"] or "").strip()
                if unver not in ("", "[]"):
                    print("    UNSAID  (unverified %s) %s" % (unver, row["to_founder"][:140]))
                elif not row["wanted_telling"]:
                    print("    unsaid  (not worth telling) %s" % row["to_founder"][:140])
                else:
                    print("    unsaid  (waiting) %s" % row["to_founder"][:140])
        return 0

    if args.command == "ledger":
        if not args.metric:
            print("Recorded metrics (%d days closed):" % ledger.days_recorded(guild_id))
            for path in ledger.LEDGER_METRICS:
                print("  %s" % path)
            print("\nPass --metric <path> for its history.")
            return 0
        if args.metric not in ledger.LEDGER_METRICS:
            print("%r is not a ledger metric. Run `ledger` with no --metric." % args.metric)
            return 1
        rows = ledger.series(guild_id, args.metric, args.days)
        if not rows:
            print("No closes recorded yet. The ledger fills in once an hour.")
            return 0
        for row in rows:
            print("%s  %s" % (row["day"], advisor.fmt_number(row["value"])))
        return 0

    if args.command == "ask":
        try:
            verdict = advisor.ask(
                " ".join(args.question), snapshot.build(guild_id), guild_id
            )
        except (advisor.Refused, advisor.BackendError) as exc:
            print(exc)
            return 1
        print(advisor.render_verdict(verdict))
        return 0

    if args.command == "brief":
        try:
            print(advisor.render_brief(advisor.brief(snapshot.build(guild_id), guild_id)))
        except (advisor.Refused, advisor.BackendError) as exc:
            print(exc)
            return 1
        return 0

    if args.command == "outcomes":
        rows = scorecard.rows_for_cli(guild_id)
        if not rows:
            print("No recommendations recorded yet for this server. Run `brief` first.")
            return 0
        # One snapshot, so an open row can show where its metric stands today
        # without waiting for its horizon. No model is involved either way.
        snap = snapshot.build(guild_id)
        print("%-6s %-10s %-34s %8s %8s %8s %8s  %s"
              % ("ref", "issued", "metric", "baseline", "current", "delta", "p", "verdict"))
        for r in rows:
            current = r["current"]
            if current is None and r["metric"] and r["metric"] != "none":
                current = snapshot.resolve_metric(snap, r["metric"])
            delta = None
            if current is not None and r["baseline"] is not None:
                delta = current - r["baseline"]
            # No "(revoked)" suffix: revocation is a verdict of its own now, and
            # the row no longer says `worked` beside it.
            verdict = r["verdict"] or "open"
            print(
                "%-6s %-10s %-34s %8s %8s %8s %8s  %s"
                % (
                    r["ref"],
                    (r["created_at"] or "")[:10],
                    (r["metric"] or "none")[:34],
                    advisor.fmt_number(r["baseline"]),
                    advisor.fmt_number(current),
                    advisor.fmt_number(delta),
                    "%.3f" % r["p_value"] if r["p_value"] is not None else "n/a",
                    verdict,
                )
            )
            print("       %s" % r["action_text"][:100])
            if r["enactment_evidence"]:
                print("       enacted: %s" % r["enactment_evidence"])
        return 0

    if args.command == "score":
        # Deterministic. This is the one command that changes what cadybot
        # believes about its own advice, and it never asks a model anything.
        closed = scorecard.score(guild_id)
        if not closed:
            print("Nothing was due for scoring.")
            return 0
        for r in closed:
            print(
                "%s  %-14s %s: baseline %s -> %s%s"
                % (
                    r["ref"],
                    r["verdict"],
                    r["metric"],
                    advisor.fmt_number(r["baseline"]),
                    advisor.fmt_number(r["current"]),
                    ", p=%.3f" % r["p_value"] if r["p_value"] is not None else "",
                )
            )
            if r.get("note"):
                print("    %s" % r["note"])
        return 0

    if args.command == "loop":
        try:
            text = asyncio.run(loop.nightly(None, guild_id))
        except (advisor.Refused, advisor.BackendError) as exc:
            print(exc)
            return 1
        print(text if text else "Nothing due and nothing moved — stayed quiet.")
        return 0

    if args.command == "purge":
        removed = db.purge_member(guild_id, args.member)
        print("Deleted %d rows for %d." % (removed, args.member))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
