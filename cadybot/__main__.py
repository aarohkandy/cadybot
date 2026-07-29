"""Command line entry point: python -m cadybot <command>"""

import argparse
import json
import sys

import discord

from . import advisor, backfill, config, db, listener, snapshot


def _one_shot_backfill() -> None:
    """Connect, import history, disconnect."""
    config.require_discord()

    intents = listener.INTENTS
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        guild = client.get_guild(config.GUILD_ID)
        if guild is None:
            print("Not in guild %s." % config.GUILD_ID)
        else:
            db.upsert_guild(guild.id, guild.name)
            total = await backfill.run(guild)
            print("Imported %d messages." % total)
        await client.close()

    client.run(config.DISCORD_TOKEN, log_handler=None)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="cadybot")
    sub = parser.add_subparsers(dest="command", required=True)

    listen = sub.add_parser("listen", help="run the gateway listener (do this first)")
    listen.add_argument(
        "--backfill", action="store_true", help="import history once on startup"
    )

    sub.add_parser("backfill", help="import history and exit")
    sub.add_parser("snapshot", help="print the raw numbers, no LLM")
    sub.add_parser("brief", help="ranked recommendations")
    sub.add_parser("outcomes", help="past recommendations and their outcomes")

    ask = sub.add_parser("ask", help="a straight yes / no / not-yet")
    ask.add_argument("question", nargs="+")

    purge = sub.add_parser("purge", help="delete one member's data")
    purge.add_argument("--member", type=int, required=True, help="Discord user id")

    args = parser.parse_args(argv)

    if args.command == "listen":
        listener.run(backfill_on_start=args.backfill)
        return 0

    if args.command == "backfill":
        _one_shot_backfill()
        return 0

    if args.command == "snapshot":
        print(json.dumps(snapshot.build(), indent=2, default=str))
        return 0

    if args.command == "ask":
        try:
            verdict = advisor.ask(" ".join(args.question), snapshot.build())
        except advisor.Refused as exc:
            print(exc)
            return 1
        print(advisor.render_verdict(verdict))
        return 0

    if args.command == "brief":
        try:
            print(advisor.render_brief(advisor.brief(snapshot.build())))
        except advisor.Refused as exc:
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
