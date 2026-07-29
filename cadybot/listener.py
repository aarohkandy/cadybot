"""The Discord client.

Joins, leaves, and voice presence exist only as live gateway events — they cannot
be recovered from Discord after the fact. That makes this the one part of cadybot
worth running before anything else is built.

Message content is never written to logs, only IDs and counts.
"""

import asyncio
import traceback
from datetime import time as dtime
from typing import Dict, Optional

import discord
from discord.ext import tasks

from . import advisor, backfill, config, db, notify, snapshot

INTENTS = discord.Intents.none()
INTENTS.guilds = True
INTENTS.members = True          # privileged: Server Members Intent
INTENTS.message_content = True  # privileged: Message Content Intent
INTENTS.guild_messages = True
INTENTS.dm_messages = True
INTENTS.guild_reactions = True
INTENTS.voice_states = True

HELP = (
    "**cadybot**\n"
    "`ask <question>` — a straight yes / no / not-yet\n"
    "`brief` — what to do this week\n"
    "`snapshot` — the raw numbers, no interpretation\n"
    "`backfill` — re-import history"
)


class Cadybot(discord.Client):
    def __init__(self, backfill_on_start: bool = False):
        super().__init__(intents=INTENTS)
        self.backfill_on_start = backfill_on_start
        self._invite_uses: Dict[str, int] = {}

    # --- lifecycle ---------------------------------------------------------

    async def on_ready(self) -> None:
        guild = self.get_guild(config.GUILD_ID)
        if guild is None:
            print("Not in guild %s. Invite the bot first." % config.GUILD_ID)
            await self.close()
            return

        db.upsert_guild(guild.id, guild.name)
        for member in guild.members:
            db.upsert_member(
                guild.id,
                member.id,
                member.name,
                member.display_name,
                member.bot,
                db.iso(member.joined_at),
            )
        for channel in guild.channels:
            db.upsert_channel(
                guild.id,
                channel.id,
                getattr(channel, "name", None),
                type(channel).__name__,
                getattr(getattr(channel, "category", None), "id", None),
                db.iso(getattr(channel, "created_at", None)),
            )
        await self._refresh_invites(guild)

        print("cadybot listening to %s (%d members)" % (guild.name, guild.member_count))

        if self.backfill_on_start:
            print("Backfilling history...")
            total = await backfill.run(guild)
            print("Backfill complete: %d messages." % total)

        if not self.weekly_brief.is_running():
            self.weekly_brief.start()
        if not self.daily_alerts.is_running():
            self.daily_alerts.start()

    async def _refresh_invites(self, guild: discord.Guild) -> None:
        """Cache invite use-counts so a join can be attributed to an invite.

        Needs Manage Server. Without it, invite attribution is simply absent —
        everything else still works.
        """
        try:
            invites = await guild.invites()
        except (discord.Forbidden, discord.HTTPException):
            return
        payload = [
            {"code": i.code, "uses": i.uses or 0, "inviter_id": getattr(i.inviter, "id", None)}
            for i in invites
        ]
        db.snapshot_invites(guild.id, payload)
        self._invite_uses = {i["code"]: i["uses"] for i in payload}

    # --- ingest ------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            if message.author.id == config.OWNER_ID:
                await self._command(message)
            return

        if message.guild.id != config.GUILD_ID:
            return

        db.upsert_member(
            message.guild.id,
            message.author.id,
            message.author.name,
            message.author.display_name,
            message.author.bot,
            db.iso(getattr(message.author, "joined_at", None)),
        )
        db.upsert_message(
            {
                "guild_id": message.guild.id,
                "channel_id": message.channel.id,
                "message_id": message.id,
                "author_id": message.author.id,
                "created_at": db.iso(message.created_at),
                "content": message.content or None,
                "reply_to_id": message.reference.message_id if message.reference else None,
                "attachments": len(message.attachments),
                "reactions": 0,
            }
        )

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id == config.GUILD_ID:
            db.bump_reactions(payload.guild_id, payload.message_id, 1)

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id == config.GUILD_ID:
            db.bump_reactions(payload.guild_id, payload.message_id, -1)

    async def on_member_join(self, member: discord.Member) -> None:
        if member.guild.id != config.GUILD_ID:
            return
        db.upsert_member(
            member.guild.id,
            member.id,
            member.name,
            member.display_name,
            member.bot,
            db.iso(member.joined_at),
        )
        code = await self._which_invite(member.guild)
        db.record_member_event(member.guild.id, member.id, "join", code)
        db.attribute_invite(member.guild.id, member.id, code)

    async def _which_invite(self, guild: discord.Guild) -> Optional[str]:
        """Whichever invite's use-count went up is the one they came through."""
        before = dict(self._invite_uses)
        await self._refresh_invites(guild)
        for code, uses in self._invite_uses.items():
            if uses > before.get(code, 0):
                return code
        return None

    async def on_member_remove(self, member: discord.Member) -> None:
        if member.guild.id != config.GUILD_ID:
            return
        db.mark_left(member.guild.id, member.id)
        db.record_member_event(member.guild.id, member.id, "leave")

    async def on_voice_state_update(self, member, before, after) -> None:
        """Presence counts only. cadybot never joins, records, or transcribes."""
        if member.guild.id != config.GUILD_ID:
            return
        if before.channel is None and after.channel is not None:
            db.open_voice(member.guild.id, after.channel.id, member.id)
        elif before.channel is not None and after.channel is None:
            db.close_voice(member.guild.id, member.id)
        elif before.channel and after.channel and before.channel.id != after.channel.id:
            db.close_voice(member.guild.id, member.id)
            db.open_voice(member.guild.id, after.channel.id, member.id)

    # --- owner commands over DM -------------------------------------------

    async def _command(self, message: discord.Message) -> None:
        raw = (message.content or "").strip()
        lowered = raw.lower()
        channel = message.channel

        try:
            if lowered.startswith("ask"):
                question = raw[3:].strip(" :")
                if not question:
                    await notify.send(channel, "Ask me something. `ask would a weekly event help?`")
                    return
                async with channel.typing():
                    snap = snapshot.build()
                    verdict = await asyncio.to_thread(advisor.ask, question, snap)
                await notify.send(channel, advisor.render_verdict(verdict))

            elif lowered.startswith("brief"):
                async with channel.typing():
                    snap = snapshot.build()
                    result = await asyncio.to_thread(advisor.brief, snap)
                await notify.send(channel, advisor.render_brief(result))

            elif lowered.startswith("snapshot"):
                import json

                snap = snapshot.build()
                await notify.send(channel, "```json\n%s\n```" % json.dumps(snap, indent=2, default=str))

            elif lowered.startswith("backfill"):
                guild = self.get_guild(config.GUILD_ID)
                await notify.send(channel, "Backfilling...")
                total = await backfill.run(guild)
                await notify.send(channel, "Imported %d messages." % total)

            else:
                await notify.send(channel, HELP)

        except advisor.Refused as exc:
            await notify.send(channel, str(exc))
        except Exception:
            traceback.print_exc()
            await notify.send(channel, "That failed. Check the listener log.")

    # --- schedules ---------------------------------------------------------

    @tasks.loop(time=dtime(hour=14, minute=0))  # 14:00 UTC Mondays
    async def weekly_brief(self) -> None:
        import datetime

        if datetime.datetime.now(datetime.timezone.utc).weekday() != 0:
            return
        try:
            snap = snapshot.build()
            result = await asyncio.to_thread(advisor.brief, snap)
            await notify.deliver(self, "**Weekly brief**\n\n" + advisor.render_brief(result))
        except Exception:
            traceback.print_exc()

    @tasks.loop(hours=24)
    async def daily_alerts(self) -> None:
        """Only speaks when something is actually wrong. No news, no message."""
        try:
            pending = snapshot.unanswered_questions(config.GUILD_ID, config.OWNER_ID)
            if not pending:
                return
            lines = ["**%d unanswered question(s).**" % len(pending), ""]
            for q in pending[:5]:
                lines.append(
                    "%s in #%s, %s days ago: %s\n%s"
                    % (q["author"], q["channel"], q["asked_days_ago"], q["text"][:180], q["link"])
                )
            await notify.deliver(self, "\n\n".join(lines), also_staff=False)
        except Exception:
            traceback.print_exc()

    @weekly_brief.before_loop
    async def _wait_weekly(self) -> None:
        await self.wait_until_ready()

    @daily_alerts.before_loop
    async def _wait_daily(self) -> None:
        await self.wait_until_ready()


def run(backfill_on_start: bool = False) -> None:
    config.require_discord()
    Cadybot(backfill_on_start=backfill_on_start).run(
        config.DISCORD_TOKEN, log_handler=None
    )
