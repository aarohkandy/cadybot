"""The Discord client.

cadybot works in every server it is added to, and keeps them completely
separate: one private channel per server, one owner per server, and every row
in the database keyed by guild_id. A test server and a live server never see
each other's numbers.

Joins, leaves, and voice presence exist only as live gateway events — they
cannot be recovered from Discord after the fact. That makes this the one part
of cadybot worth running before anything else is built.

Message content is never written to logs, only IDs and counts.
"""

import asyncio
import json
import traceback
from datetime import time as dtime
from typing import Dict, Optional

import discord
from discord import app_commands
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

NO_ROOM = "No private channel in this server yet. Run `/private` first."


class Cadybot(discord.Client):
    def __init__(self, backfill_on_start: bool = False):
        super().__init__(intents=INTENTS)
        self.tree = app_commands.CommandTree(self)
        self.backfill_on_start = backfill_on_start
        self._rooms: Dict[int, Optional[int]] = {}
        self._talking: Dict[int, asyncio.Lock] = {}
        self._invite_uses: Dict[int, Dict[str, int]] = {}

    # --- lifecycle ---------------------------------------------------------

    async def setup_hook(self) -> None:
        register_commands(self)

    async def on_ready(self) -> None:
        print("cadybot online as %s" % self.user)
        print("backend: %s" % llm.describe())

        # Commands are registered per-guild (instant) rather than globally (up
        # to an hour to propagate). Doing both makes Discord list every command
        # twice, so drop any global copies left over from an earlier run. The
        # in-memory tree is untouched, so copy_global_to still works below.
        try:
            await self.http.bulk_upsert_global_commands(self.application_id, [])
        except Exception as exc:
            print("could not clear global commands: %s" % exc)

        problem = llm.preflight()
        if problem:
            print("WARNING: %s" % problem)
        else:
            # Load the model now rather than making the first question wait for
            # it. Runs in a thread so the gateway connection isn't blocked.
            asyncio.get_event_loop().run_in_executor(None, llm.warm)

        if not self.guilds:
            print("Not in any server yet. Use `python -m cadybot invite`.")

        for guild in self.guilds:
            await self._adopt(guild)

        if not self.weekly_brief.is_running():
            self.weekly_brief.start()
        if not self.daily_alerts.is_running():
            self.daily_alerts.start()

    async def _adopt(self, guild: discord.Guild) -> None:
        """Register a server and sync its slash commands. Safe to repeat."""
        db.upsert_guild(guild.id, guild.name)
        for member in guild.members:
            db.upsert_member(
                guild.id, member.id, member.name, member.display_name,
                member.bot, db.iso(member.joined_at),
            )
        for channel in guild.channels:
            db.upsert_channel(
                guild.id, channel.id, getattr(channel, "name", None),
                type(channel).__name__,
                getattr(getattr(channel, "category", None), "id", None),
                db.iso(getattr(channel, "created_at", None)),
            )
        await self._refresh_invites(guild)
        self._rooms[guild.id] = room.stored_id(guild.id)

        # Guild-scoped sync appears immediately, unlike the global sync.
        try:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
        except discord.HTTPException as exc:
            print("  command sync failed for %s: %s" % (guild.name, exc))

        state = "#%s" % self._rooms[guild.id] if self._rooms.get(guild.id) else "no /private yet"
        print("  %s (%s, %d members) — %s" % (guild.name, guild.id, guild.member_count, state))

        if self.backfill_on_start:
            total = await backfill.run(guild, skip_channel_id=self._rooms.get(guild.id))
            print("    backfilled %d messages" % total)

    async def on_guild_join(self, guild: discord.Guild) -> None:
        print("added to %s (%s)" % (guild.name, guild.id))
        await self._adopt(guild)

    async def on_guild_channel_delete(self, channel) -> None:
        guild = getattr(channel, "guild", None)
        if guild and self._rooms.get(guild.id) == channel.id:
            room.forget(guild.id)
            self._rooms[guild.id] = None

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
        self._invite_uses[guild.id] = {i["code"]: i["uses"] for i in payload}

    # --- ingest ------------------------------------------------------------

    async def on_message(self, message: discord.Message) -> None:
        if message.author.id == self.user.id or message.guild is None:
            return

        # The private channel is a console, not part of the server's life. Its
        # messages are never stored as server activity, so they can never
        # inflate the snapshot — they are a conversation instead.
        if message.channel.id == self._rooms.get(message.guild.id):
            await self._converse(message)
            return

        db.upsert_member(
            message.guild.id, message.author.id, message.author.name,
            message.author.display_name, message.author.bot,
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
        if payload.guild_id and payload.channel_id != self._rooms.get(payload.guild_id):
            db.bump_reactions(payload.guild_id, payload.message_id, 1)

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent) -> None:
        if payload.guild_id and payload.channel_id != self._rooms.get(payload.guild_id):
            db.bump_reactions(payload.guild_id, payload.message_id, -1)

    async def on_member_join(self, member: discord.Member) -> None:
        db.upsert_member(
            member.guild.id, member.id, member.name, member.display_name,
            member.bot, db.iso(member.joined_at),
        )
        code = await self._which_invite(member.guild)
        db.record_member_event(member.guild.id, member.id, "join", code)
        db.attribute_invite(member.guild.id, member.id, code)

    async def _which_invite(self, guild: discord.Guild) -> Optional[str]:
        """Whichever invite's use-count went up is the one they came through."""
        before = dict(self._invite_uses.get(guild.id, {}))
        await self._refresh_invites(guild)
        for code, uses in self._invite_uses.get(guild.id, {}).items():
            if uses > before.get(code, 0):
                return code
        return None

    async def on_member_remove(self, member: discord.Member) -> None:
        db.mark_left(member.guild.id, member.id)
        db.record_member_event(member.guild.id, member.id, "leave")

    async def on_voice_state_update(self, member, before, after) -> None:
        """Presence counts only. cadybot never joins, records, or transcribes."""
        gid = member.guild.id
        if before.channel is None and after.channel is not None:
            db.open_voice(gid, after.channel.id, member.id)
        elif before.channel is not None and after.channel is None:
            db.close_voice(gid, member.id)
        elif before.channel and after.channel and before.channel.id != after.channel.id:
            db.close_voice(gid, member.id)
            db.open_voice(gid, after.channel.id, member.id)

    # --- conversation ------------------------------------------------------

    async def _converse(self, message: discord.Message) -> None:
        """Talk back. Any ordinary message in the private channel is a question.

        Prefix a line with // to say something without cadybot answering, so the
        channel is still usable for notes and side chat between people.
        """
        if message.author.bot:
            return
        text = (message.content or "").strip()
        if not text or text.startswith("//"):
            return

        lock = self._talking.setdefault(message.channel.id, asyncio.Lock())
        if lock.locked():
            # Still thinking about the previous message. Say so once rather than
            # silently queueing a reply that arrives minutes later out of order.
            await message.add_reaction("\N{HOURGLASS WITH FLOWING SAND}")
            return

        async with lock:
            try:
                async with message.channel.typing():
                    snap = snapshot.build(message.guild.id)
                    reply = await asyncio.to_thread(
                        advisor.chat,
                        message.guild.id,
                        message.channel.id,
                        text,
                        message.author.display_name,
                        snap,
                    )
                if reply:
                    await notify.send(message.channel, reply)
            except (advisor.Refused, advisor.BackendError) as exc:
                await notify.send(message.channel, str(exc))
            except Exception:
                traceback.print_exc()
                await notify.send(message.channel, "That failed. Check the listener log.")

    # --- helpers used by the commands --------------------------------------

    def room_channel(self, guild: discord.Guild) -> Optional[discord.TextChannel]:
        channel_id = self._rooms.get(guild.id) or room.stored_id(guild.id)
        if channel_id is None:
            return None
        self._rooms[guild.id] = channel_id
        return guild.get_channel(channel_id)

    def may_read(self, interaction: discord.Interaction) -> bool:
        """Analytics are for people who can see the private channel."""
        member = interaction.user
        if getattr(member.guild_permissions, "manage_guild", False):
            return True
        channel = self.room_channel(interaction.guild)
        if channel is None:
            return False
        return channel.permissions_for(member).view_channel

    def in_room(self, interaction: discord.Interaction) -> bool:
        return interaction.channel_id == self._rooms.get(interaction.guild_id)

    # --- schedules ---------------------------------------------------------

    @tasks.loop(time=dtime(hour=14, minute=0))  # 14:00 UTC Mondays
    async def weekly_brief(self) -> None:
        import datetime

        if datetime.datetime.now(datetime.timezone.utc).weekday() != 0:
            return
        for guild in list(self.guilds):
            if not self._rooms.get(guild.id):
                continue
            try:
                snap = snapshot.build(guild.id)
                result = await asyncio.to_thread(advisor.brief, snap, guild.id)
                await notify.deliver(
                    self, guild.id, "**Weekly brief**\n\n" + advisor.render_brief(result)
                )
            except Exception:
                traceback.print_exc()

    @tasks.loop(hours=24)
    async def daily_alerts(self) -> None:
        """Only speaks when something is actually wrong. No news, no message."""
        for guild in list(self.guilds):
            if not self._rooms.get(guild.id):
                continue
            try:
                pending = snapshot.unanswered_questions(guild.id, room.owner_id(guild.id) or 0)
                if not pending:
                    continue
                lines = ["**%d unanswered question(s).**" % len(pending), ""]
                for q in pending[:5]:
                    lines.append(
                        "%s in #%s, %s days ago: %s\n%s"
                        % (q["author"], q["channel"], q["asked_days_ago"],
                           q["text"][:180], q["link"])
                    )
                await notify.deliver(self, guild.id, "\n\n".join(lines))
            except Exception:
                traceback.print_exc()

    @weekly_brief.before_loop
    async def _wait_weekly(self) -> None:
        await self.wait_until_ready()

    @daily_alerts.before_loop
    async def _wait_daily(self) -> None:
        await self.wait_until_ready()


# --- slash commands --------------------------------------------------------


async def _reply(interaction: discord.Interaction, text: str, ephemeral: bool) -> None:
    """Send a possibly-long reply, chunked to Discord's 2000-character limit.

    An interaction token dies if the bot restarts mid-command or the reply takes
    longer than Discord's window — both plausible when a 9GB local model is
    doing the thinking. Rather than lose an answer that cost real time to
    produce, fall back to posting it in the private channel.
    """
    parts = notify.chunk(text)
    try:
        for part in parts:
            await interaction.followup.send(part, ephemeral=ephemeral)
        return
    except discord.HTTPException as exc:
        print("interaction reply failed (%s); falling back to the channel" % exc)

    guild_id = interaction.guild_id
    if guild_id:
        try:
            await notify.deliver(
                interaction.client,
                guild_id,
                "%s\n%s" % (interaction.user.mention, text),
            )
        except Exception:
            traceback.print_exc()


def register_commands(bot: Cadybot) -> None:
    tree = bot.tree

    @tree.command(name="private", description="Create your private cadybot channel in this server")
    @app_commands.guild_only()
    @app_commands.checks.has_permissions(manage_guild=True)
    async def private(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        try:
            channel = await room.create(interaction.guild, interaction.user)
        except room.RoomError as exc:
            await _reply(interaction, str(exc), True)
            return
        bot._rooms[interaction.guild_id] = channel.id
        await _reply(
            interaction,
            "Done — %s is yours. Only you can see it; use `/add` to let others in."
            % channel.mention,
            True,
        )

    @tree.command(name="ask", description="A straight yes / no / not-yet on one idea")
    @app_commands.guild_only()
    @app_commands.describe(question="e.g. would a weekly event help?")
    async def ask(interaction: discord.Interaction, question: str) -> None:
        if not bot.may_read(interaction):
            await interaction.response.send_message(NO_ROOM, ephemeral=True)
            return
        visible = bot.in_room(interaction)
        await interaction.response.defer(ephemeral=not visible)
        try:
            snap = snapshot.build(interaction.guild_id)
            verdict = await asyncio.to_thread(advisor.ask, question, snap, interaction.guild_id)
        except (advisor.Refused, advisor.BackendError) as exc:
            await _reply(interaction, str(exc), True)
            return
        await _reply(
            interaction,
            "> %s\n\n%s" % (question, advisor.render_verdict(verdict)),
            not visible,
        )

    @tree.command(name="brief", description="What to do this week, ranked")
    @app_commands.guild_only()
    async def brief(interaction: discord.Interaction) -> None:
        if not bot.may_read(interaction):
            await interaction.response.send_message(NO_ROOM, ephemeral=True)
            return
        visible = bot.in_room(interaction)
        await interaction.response.defer(ephemeral=not visible)
        try:
            snap = snapshot.build(interaction.guild_id)
            result = await asyncio.to_thread(advisor.brief, snap, interaction.guild_id)
        except (advisor.Refused, advisor.BackendError) as exc:
            await _reply(interaction, str(exc), True)
            return
        await _reply(interaction, advisor.render_brief(result), not visible)

    @tree.command(name="snapshot", description="The raw numbers, no interpretation")
    @app_commands.guild_only()
    async def snap_cmd(interaction: discord.Interaction) -> None:
        if not bot.may_read(interaction):
            await interaction.response.send_message(NO_ROOM, ephemeral=True)
            return
        visible = bot.in_room(interaction)
        await interaction.response.defer(ephemeral=not visible)
        body = json.dumps(snapshot.build(interaction.guild_id), indent=2, default=str)
        await _reply(interaction, "```json\n%s\n```" % body, not visible)

    @tree.command(name="who", description="Who can see the private channel")
    @app_commands.guild_only()
    async def who(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        channel = bot.room_channel(interaction.guild)
        if channel is None:
            await _reply(interaction, NO_ROOM, True)
            return
        seats = room.roster(channel)
        lines = [
            "**%s**" % channel.mention,
            "",
            "**Added:** %s" % (", ".join(seats["added"]) or "just cadybot"),
        ]
        if seats["bypassing"]:
            lines += [
                "",
                "**Can also read it anyway:**",
                "\n".join("- " + s for s in seats["bypassing"]),
                "",
                "Administrators bypass channel privacy — `/remove` cannot shut them "
                "out. Take Administrator off their role in Server Settings → Roles "
                "if you need this channel actually sealed.",
            ]
        await _reply(interaction, "\n".join(lines), True)

    @tree.command(name="add", description="Let someone into the private channel")
    @app_commands.guild_only()
    @app_commands.checks.has_permissions(manage_guild=True)
    async def add(interaction: discord.Interaction, member: discord.Member) -> None:
        await interaction.response.defer(ephemeral=True)
        channel = bot.room_channel(interaction.guild)
        if channel is None:
            await _reply(interaction, NO_ROOM, True)
            return
        try:
            await room.add(channel, member)
        except room.RoomError as exc:
            await _reply(interaction, str(exc), True)
            return
        await _reply(interaction, "Added %s to %s." % (member.display_name, channel.mention), True)

    @tree.command(name="remove", description="Remove someone from the private channel")
    @app_commands.guild_only()
    @app_commands.checks.has_permissions(manage_guild=True)
    async def remove(interaction: discord.Interaction, member: discord.Member) -> None:
        await interaction.response.defer(ephemeral=True)
        channel = bot.room_channel(interaction.guild)
        if channel is None:
            await _reply(interaction, NO_ROOM, True)
            return
        try:
            await room.remove(channel, member)
        except room.RoomError as exc:
            await _reply(interaction, str(exc), True)
            return
        await _reply(
            interaction, "Removed %s from %s." % (member.display_name, channel.mention), True
        )

    @tree.command(name="backfill", description="Import this server's message history")
    @app_commands.guild_only()
    @app_commands.checks.has_permissions(manage_guild=True)
    async def backfill_cmd(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send("Importing history — this can take a while.", ephemeral=True)
        total = await backfill.run(
            interaction.guild, skip_channel_id=bot._rooms.get(interaction.guild_id)
        )
        await _reply(interaction, "Imported %d messages." % total, True)

    @tree.command(name="reset", description="Forget the conversation so far in this channel")
    @app_commands.guild_only()
    async def reset(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        if not bot.may_read(interaction):
            await _reply(interaction, NO_ROOM, True)
            return
        n = db.clear_turns(interaction.guild_id, interaction.channel_id)
        await _reply(interaction, "Forgot %d turns. Starting fresh." % n, True)

    @tree.error
    async def on_error(interaction: discord.Interaction, error: Exception) -> None:
        if isinstance(error, app_commands.MissingPermissions):
            message = "You need **Manage Server** in this server to do that."
        elif isinstance(error, app_commands.NoPrivateMessage):
            message = "Run this inside a server, not in a DM."
        else:
            traceback.print_exc()
            message = "That failed. Check the listener log."
        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except discord.HTTPException:
            pass


def run(backfill_on_start: bool = False) -> None:
    config.require_discord()
    Cadybot(backfill_on_start=backfill_on_start).run(config.DISCORD_TOKEN, log_handler=None)
