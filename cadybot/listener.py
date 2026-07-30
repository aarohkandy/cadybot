"""The Discord client.

Joins, leaves, and voice presence exist only as live gateway events — they
cannot be recovered from Discord after the fact. That makes this the one part of
cadybot worth running before anything else is built.

Message content is never written to logs, only IDs and counts.
"""

import asyncio
import json
import traceback
from datetime import time as dtime
from typing import Dict, Optional

import discord
from discord.ext import tasks

from . import advisor, backfill, config, db, llm, notify, room, snapshot

INTENTS = discord.Intents.none()
INTENTS.guilds = True
INTENTS.members = True          # privileged: Server Members Intent
INTENTS.message_content = True  # privileged: Message Content Intent
INTENTS.guild_messages = True
INTENTS.dm_messages = True
INTENTS.guild_reactions = True
INTENTS.voice_states = True

HELP = (
    "`ask <question>` — a straight yes / no / not-yet\n"
    "`brief` — what to do this week\n"
    "`snapshot` — the raw numbers, no interpretation\n"
    "`who` — who can see this channel\n"
    "`add @someone` / `remove @someone` — change that (owner only)\n"
    "`backfill` — re-import history (owner only)"
)

KNOWN = ("ask", "brief", "snapshot", "who", "add", "remove", "backfill")
OWNER_ONLY = ("add", "remove", "backfill")


class Cadybot(discord.Client):
    def __init__(self, backfill_on_start: bool = False):
        super().__init__(intents=INTENTS)
        self.backfill_on_start = backfill_on_start
        self._invite_uses: Dict[str, int] = {}
        self._room_id: Optional[int] = None

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

        try:
            channel = await room.ensure(guild)
            self._room_id = channel.id
            print("private channel: #%s (%s)" % (channel.name, channel.id))
        except room.RoomError as exc:
            print("Could not set up the private channel: %s" % exc)
            print("Falling back to DMs.")

        print("cadybot listening to %s (%d members)" % (guild.name, guild.member_count))
        print("backend: %s" % llm.describe())

        problem = llm.preflight()
        if problem:
            print("WARNING: %s" % problem)
            await notify.deliver(self, "Backend problem: %s" % problem)

        if self.backfill_on_start:
            print("Backfilling history...")
            total = await backfill.run(guild, skip_channel_id=self._room_id)
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
        if message.author.id == self.user.id:
            return

        if message.guild is None:
            if message.author.id == config.OWNER_ID:
                await self._command(message)
            return

        if message.guild.id != config.GUILD_ID:
            return

        # The private channel is a console, not part of the server's life. Its
        # messages are never stored, so they can never inflate the snapshot.
        if message.channel.id == self._room_id:
            await self._command(message)
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
        if payload.guild_id == config.GUILD_ID and payload.channel_id != self._room_id:
            db.bump_reactions(payload.guild_id, payload.message_id, 1)

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id == config.GUILD_ID and payload.channel_id != self._room_id:
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

    # --- commands ----------------------------------------------------------

    async def _command(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        raw = (message.content or "").strip()
        if not raw:
            return

        verb = raw.split(" ", 1)[0].lower()
        rest = raw[len(verb):].strip(" :")
        channel = message.channel
        is_owner = message.author.id == config.OWNER_ID

        if verb in ("help", "h", "?", "commands"):
            await notify.send(channel, HELP)
            return

        # Anything that isn't a command is left alone, so the channel stays
        # usable as an actual conversation rather than a command prompt that
        # barks the help text at every message.
        if verb not in KNOWN:
            return

        if verb in OWNER_ONLY and not is_owner:
            await notify.send(channel, "Only the owner can do that.")
            return

        try:
            if verb == "ask":
                if not rest:
                    await notify.send(channel, "Ask me something. `ask would a weekly event help?`")
                    return
                async with channel.typing():
                    snap = snapshot.build()
                    verdict = await asyncio.to_thread(advisor.ask, rest, snap)
                await notify.send(channel, advisor.render_verdict(verdict))

            elif verb == "brief":
                async with channel.typing():
                    snap = snapshot.build()
                    result = await asyncio.to_thread(advisor.brief, snap)
                await notify.send(channel, advisor.render_brief(result))

            elif verb == "snapshot":
                snap = snapshot.build()
                await notify.send(
                    channel, "```json\n%s\n```" % json.dumps(snap, indent=2, default=str)
                )

            elif verb == "who":
                await self._who(message)

            elif verb in ("add", "remove"):
                await self._membership(message, verb)

            elif verb == "backfill":
                guild = self.get_guild(config.GUILD_ID)
                await notify.send(channel, "Backfilling...")
                total = await backfill.run(guild, skip_channel_id=self._room_id)
                await notify.send(channel, "Imported %d messages." % total)

        except (advisor.Refused, advisor.BackendError, room.RoomError) as exc:
            await notify.send(channel, str(exc))
        except Exception:
            traceback.print_exc()
            await notify.send(channel, "That failed. Check the listener log.")

    async def _room_channel(self) -> Optional[discord.TextChannel]:
        guild = self.get_guild(config.GUILD_ID)
        if guild is None or self._room_id is None:
            return None
        return guild.get_channel(self._room_id)

    async def _who(self, message: discord.Message) -> None:
        channel = await self._room_channel()
        if channel is None:
            await notify.send(message.channel, "No private channel set up yet.")
            return
        names = room.roster(channel)
        await notify.send(
            message.channel,
            "Can see #%s: %s" % (channel.name, ", ".join(names) if names else "just cadybot"),
        )

    async def _membership(self, message: discord.Message, verb: str) -> None:
        channel = await self._room_channel()
        if channel is None:
            await notify.send(message.channel, "No private channel set up yet.")
            return

        targets = list(message.mentions)
        if not targets:
            await notify.send(message.channel, "Mention someone: `%s @user`" % verb)
            return

        guild = channel.guild
        done = []
        for user in targets:
            member = guild.get_member(user.id)
            if member is None:
                try:
                    member = await guild.fetch_member(user.id)
                except discord.NotFound:
                    await notify.send(message.channel, "%s isn't in this server." % user)
                    continue
            if verb == "add":
                await room.add(channel, member)
            else:
                await room.remove(channel, member)
            done.append(member.display_name)

        if done:
            await notify.send(
                message.channel,
                "%s %s." % ("Added" if verb == "add" else "Removed", ", ".join(done)),
            )

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
            await notify.deliver(self, "\n\n".join(lines))
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
