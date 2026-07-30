"""Command line entry point: python -m cadybot <command>"""

import argparse
import json
import sys

import discord

from . import advisor, backfill, config, db, listener, llm, room, snapshot


def _one_shot_backfill() -> None:
    """Connect, import history, disconnect."""
    config.require_discord()

    client = discord.Client(intents=listener.INTENTS)

    @client.event
    async def on_ready():
        guild = client.get_guild(config.GUILD_ID)
        if guild is None:
            print("Not in guild %s." % config.GUILD_ID)
        else:
            db.upsert_guild(guild.id, guild.name)
            total = await backfill.run(guild, skip_channel_id=room.stored_id(guild.id))
            print("Imported %d messages." % total)
        await client.close()

    client.run(config.DISCORD_TOKEN, log_handler=None)


# View Channels | Send Messages | Read Message History | Manage Channels
# | Manage Roles | Manage Server. Manage Channels/Roles are what let cadybot
# create its private channel and control who sees it; Manage Server is what
# lets it read invites for join attribution.
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

    url = "https://discord.com/oauth2/authorize?client_id=%s&scope=bot&permissions=%d"
    print("Everything cadybot needs, and nothing more:")
    print("  " + url % (app_id, SCOPED_PERMS))
    print()
    print("Administrator (test servers only — grants far more than it uses):")
    print("  " + url % (app_id, ADMIN_PERMS))
    print()
    print("Enable Server Members Intent and Message Content Intent under")
    print("Developer Portal -> your app -> Bot -> Privileged Gateway Intents.")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cadybot")
    sub = parser.add_subparsers(dest="command", required=True)

    listen = sub.add_parser("listen", help="run the gateway listener (do this first)")
    listen.add_argument(
        "--backfill", action="store_true", help="import history once on startup"
    )

    sub.add_parser("invite", help="print the OAuth invite URL")
    sub.add_parser("backfill", help="import history and exit")
    sub.add_parser("snapshot", help="print the raw numbers, no LLM")
    sub.add_parser("brief", help="ranked recommendations")
    sub.add_parser("outcomes", help="past recommendations and their outcomes")
    sub.add_parser("doctor", help="check the model backend is reachable")

    ask = sub.add_parser("ask", help="a straight yes / no / not-yet")
    ask.add_argument("question", nargs="+")

    purge = sub.add_parser("purge", help="delete one member's data")
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

    if args.command == "backfill":
        _one_shot_backfill()
        return 0

    if args.command == "snapshot":
        print(json.dumps(snapshot.build(), indent=2, default=str))
        return 0

    if args.command == "ask":
        try:
            verdict = advisor.ask(" ".join(args.question), snapshot.build())
        except (advisor.Refused, advisor.BackendError) as exc:
            print(exc)
            return 1
        print(advisor.render_verdict(verdict))
        return 0

    if args.command == "brief":
        try:
            print(advisor.render_brief(advisor.brief(snapshot.build())))
        except (advisor.Refused, advisor.BackendError) as exc:
            print(exc)
            return 1
        return 0

    if args.command == "outcomes":
        rows = db.query(
            "SELECT created_at, headline, metric, prediction, outcome FROM recommendations "
            "WHERE guild_id=? ORDER BY created_at DESC LIMIT 30",
            (config.GUILD_ID,),
        )
        if not rows:
            print("No recommendations recorded yet. Run `brief` first.")
            return 0
        for r in rows:
            print("%s  %s" % (r["created_at"][:10], r["headline"]))
            print("    watch: %s -> %s" % (r["metric"], r["prediction"]))
            print("    outcome: %s" % (r["outcome"] or "not yet reviewed"))
        return 0

    if args.command == "purge":
        removed = db.purge_member(config.GUILD_ID, args.member)
        print("Deleted %d rows for %d." % (removed, args.member))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
