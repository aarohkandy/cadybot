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
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from typing import Dict, Optional

import discord
from discord import app_commands
from discord.ext import tasks

from . import (
    advisor, agenda, backfill, config, db, inquiry, ledger, llm, loop, notify,
    room, snapshot, thinking,
)

INTENTS = discord.Intents.none()
INTENTS.guilds = True
INTENTS.members = True          # privileged: Server Members Intent
INTENTS.message_content = True  # privileged: Message Content Intent
INTENTS.guild_messages = True
INTENTS.dm_messages = True
INTENTS.guild_reactions = True
INTENTS.voice_states = True
# GUILD_MODERATION. Not privileged and needs no approval, but a running client
# keeps whatever intents it connected with, so this takes effect on the next
# deliberate restart rather than immediately.
INTENTS.moderation = True

NO_ROOM = "No private channel in this server yet. Run `/private` first."

# The only audit-log actions worth keeping. A kick, a ban and a prune are the
# only departures Discord will ever explain; the rest of the audit log is server
# administration and says nothing about whether the community is working.
AUDIT_ACTIONS = (
    discord.AuditLogAction.kick,
    discord.AuditLogAction.member_prune,
    discord.AuditLogAction.ban,
    discord.AuditLogAction.unban,
    discord.AuditLogAction.automod_block_message,
)
# Catch-up is bounded per action per start. A busy server's audit log has no end,
# and a start that never finishes is worse than a gap.
AUDIT_CATCHUP_LIMIT = 200


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

        # hourly_facts always runs: it is ingest, not reporting, and it is the
        # heartbeat the health check reads. Only the two reporting passes are
        # handed to cron.
        if config.SCHEDULER == "cron":
            print("scheduler: cron (weekly/nightly via `cadybot post`); "
                  "desk runs in-process every %dm" % config.SCAN_INTERVAL_MINUTES)
        else:
            if not self.weekly_brief.is_running():
                self.weekly_brief.start()
            if not self.daily_alerts.is_running():
                self.daily_alerts.start()
        # The desk starts under BOTH scheduler modes, unlike the two reporting
        # passes. An idle look is 2ms of indexed SELECTs and no tokens, and it
        # is the part that has to be responsive — handing it to a timer meant
        # hours between somebody asking a question and cadybot noticing. What
        # cron still owns is the nightly and weekly reports.
        if not self.think_pass.is_running():
            self.think_pass.start()
        if not self.hourly_facts.is_running():
            self.hourly_facts.start()

    async def _adopt(self, guild: discord.Guild) -> None:
        """Register a server and sync its slash commands. Safe to repeat."""
        db.upsert_guild(guild.id, guild.name)
        for member in guild.members:
            db.upsert_member(
                guild.id, member.id, member.name, member.display_name,
                member.bot, db.iso(member.joined_at), db.member_state(member),
            )
        for channel in guild.channels:
            backfill.register_channel(guild.id, channel)
        # Guild.channels is GuildChannel only, so threads have to be walked
        # separately or a forum server looks like it has no channels at all.
        for thread in guild.threads:
            backfill.register_channel(guild.id, thread)
        await self._refresh_invites(guild)
        await self._refresh_facts(guild)
        await self._catch_up_audit(guild)
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

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        """Removed from a server means forgotten. Nothing is kept for later."""
        self._rooms.pop(guild.id, None)
        self._invite_uses.pop(guild.id, None)
        print("removed from %s (%s); purged %d rows" % (guild.name, guild.id,
                                                        db.purge_guild(guild.id)))

    async def on_guild_channel_delete(self, channel) -> None:
        guild = getattr(channel, "guild", None)
        if guild and self._rooms.get(guild.id) == channel.id:
            room.forget(guild.id)
            self._rooms[guild.id] = None

    async def on_guild_channel_create(self, channel) -> None:
        guild = getattr(channel, "guild", None)
        if guild:
            backfill.register_channel(guild.id, channel)

    async def on_guild_channel_update(self, before, after) -> None:
        guild = getattr(after, "guild", None)
        if guild:
            backfill.register_channel(guild.id, after)

    async def on_thread_create(self, thread: discord.Thread) -> None:
        # Not on_thread_join: discord.py splits THREAD_CREATE on the newly_created
        # flag, and the join half also fires when cadybot is merely added to a
        # thread that has existed for months.
        backfill.register_channel(thread.guild.id, thread)

    async def on_thread_update(self, before: discord.Thread, after: discord.Thread) -> None:
        backfill.register_channel(after.guild.id, after)

    async def _refresh_facts(self, guild: discord.Guild) -> None:
        """Server settings cadybot can see, and honesty about the ones it cannot.

        Every column left NULL means not readable. That has to stay distinct
        from 0, because telling the founder onboarding is disabled when the truth
        is that cadybot cannot read onboarding sends them to fix a setting that
        was never broken.
        """
        facts = {
            "widget_enabled": (
                None if guild.widget_enabled is None else int(guild.widget_enabled)
            ),
            "boost_count": guild.premium_subscription_count,
            "boost_tier": guild.premium_tier,
            "verification_level": str(guild.verification_level),
            "is_community": 1 if "COMMUNITY" in guild.features else 0,
        }
        # guild.onboarding() raises outside a Community server, so the feature
        # flag is the gate rather than the exception handler.
        if "COMMUNITY" in guild.features:
            try:
                onboarding = await guild.onboarding()
            except (discord.Forbidden, discord.HTTPException):
                facts["onboarding_readable"] = 0
            else:
                facts["onboarding_readable"] = 1
                facts["onboarding_enabled"] = int(onboarding.enabled)
                facts["onboarding_mode"] = str(onboarding.mode)
                facts["onboarding_prompts"] = len(onboarding.prompts)
                facts["onboarding_required_prompts"] = sum(
                    1 for p in onboarding.prompts if p.required
                )
                facts["onboarding_default_channels"] = len(onboarding.default_channel_ids)
        db.upsert_guild_facts(guild.id, facts)

    async def _catch_up_audit(self, guild: discord.Guild) -> None:
        """Bounded backfill of the moderation events missed while offline."""
        for action in AUDIT_ACTIONS:
            after = db.last_audit_entry_id(guild.id, action.value)
            kwargs = {"action": action, "oldest_first": True, "limit": AUDIT_CATCHUP_LIMIT}
            if after:
                kwargs["after"] = discord.Object(id=after)
            try:
                async for entry in guild.audit_logs(**kwargs):
                    self._store_audit(guild.id, entry)
            except discord.Forbidden:
                db.upsert_guild_facts(guild.id, {"audit_readable": 0})
                return
            except discord.HTTPException as exc:
                print("  audit catch-up failed for %s: %s" % (guild.name, exc))
                return
        db.upsert_guild_facts(guild.id, {"audit_readable": 1})

    def _store_audit(self, guild_id: int, entry) -> None:
        if entry.action not in AUDIT_ACTIONS:
            return
        # Gateway audit entries are resolved against the cache, so target is
        # usually a bare Object and occasionally nothing at all.
        db.record_audit_event(
            guild_id,
            entry.id,
            entry.action.value,
            entry.user_id,
            getattr(entry.target, "id", None),
            db.iso(entry.created_at),
            _audit_extra(entry),
        )

    async def on_audit_log_entry_create(self, entry) -> None:
        if entry.guild is not None:
            self._store_audit(entry.guild.id, entry)

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
            db.member_state(message.author),
        )
        db.upsert_message(
            backfill.message_row(message.guild.id, message.channel.id, message, 0)
        )
        # message.mentions comes from the gateway payload's mentions array, not
        # from scanning the text, so it survives losing the Message Content
        # intent. @everyone and @here are not in it and live on the message row.
        db.add_mentions(
            message.guild.id, message.id, message.author.id,
            [user.id for user in message.mentions],
            db.iso(message.created_at),
        )

    def _outside_room(self, payload) -> bool:
        """True when a raw event belongs to server life rather than the console."""
        return bool(payload.guild_id) and payload.channel_id != self._rooms.get(
            payload.guild_id
        )

    async def on_raw_message_delete(self, payload: discord.RawMessageDeleteEvent) -> None:
        # The raw variant, because on_message_delete only fires for messages
        # still in discord.py's 1000-entry cache. Under launchd the process runs
        # for weeks, so nearly every deletion is of a message that aged out.
        if self._outside_room(payload):
            db.delete_messages(payload.guild_id, [payload.message_id])

    async def on_raw_bulk_message_delete(
        self, payload: discord.RawBulkMessageDeleteEvent
    ) -> None:
        if self._outside_room(payload):
            db.delete_messages(payload.guild_id, list(payload.message_ids))

    async def on_raw_message_edit(self, payload: discord.RawMessageUpdateEvent) -> None:
        # Discord fires MESSAGE_UPDATE with no content key at all when its embed
        # server finishes unfurling a link. On a server where people trade CAD
        # models and print photos that is most update events, and acting on them
        # would stamp an edit time on messages nobody touched.
        if 'content' in payload.data and self._outside_room(payload):
            db.update_message_content(
                payload.guild_id,
                payload.message_id,
                payload.data['content'],
                edited_at=db.iso(payload.message.edited_at),
                flags=payload.message.flags.value,
            )

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        if self._outside_room(payload):
            db.add_reaction(
                payload.guild_id, payload.message_id, payload.user_id, str(payload.emoji)
            )

    async def on_raw_reaction_remove(self, payload: discord.RawReactionActionEvent) -> None:
        if self._outside_room(payload):
            db.remove_reaction(
                payload.guild_id, payload.message_id, payload.user_id, str(payload.emoji)
            )

    async def on_raw_reaction_clear(self, payload: discord.RawReactionClearEvent) -> None:
        if self._outside_room(payload):
            db.clear_reactions(payload.guild_id, payload.message_id)

    async def on_raw_reaction_clear_emoji(
        self, payload: discord.RawReactionClearEmojiEvent
    ) -> None:
        if self._outside_room(payload):
            db.clear_reactions(payload.guild_id, payload.message_id, str(payload.emoji))

    async def on_member_join(self, member: discord.Member) -> None:
        db.upsert_member(
            member.guild.id, member.id, member.name, member.display_name,
            member.bot, db.iso(member.joined_at), db.member_state(member),
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

    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        """The only event that fires when someone finishes onboarding.

        pending goes True to False exactly once and nothing else reports it — a
        member stuck pending never saw a single channel, which looks identical
        to a member who joined and chose not to speak.
        """
        db.upsert_member(
            after.guild.id, after.id, after.name, after.display_name,
            after.bot, db.iso(after.joined_at), db.member_state(after),
        )

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
                    # Look things up first, then answer. Both run on a worker
                    # thread: db.connect() is thread-local and the whole turn
                    # is minutes long on CPU inference. channel.typing()
                    # re-triggers for the duration, so the wait shows as typing
                    # rather than as dead air.
                    inq = await asyncio.to_thread(
                        inquiry.investigate, message.guild.id, text,
                        config.INQUIRY_ROUNDS_CHAT, config.INQUIRY_BUDGET_CHAT,
                    )
                    reply = await asyncio.to_thread(
                        advisor.chat,
                        message.guild.id,
                        message.channel.id,
                        text,
                        message.author.display_name,
                        snap,
                        inq,
                    )
                if reply:
                    await notify.send(message.channel, reply)
                    # A conversational reply is cadybot speaking, and the desk's
                    # "do not talk over anything else" gap is measured off this
                    # table. Without it the founder can be mid-conversation and
                    # still get an unsolicited thought dropped on top of it,
                    # because notify.send is a second door that notify.deliver's
                    # logging never covered.
                    db.record_delivery(message.guild.id, "chat", len(reply))
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
                # loop.weekly rather than advisor.brief directly: grading has to
                # happen, and has to be committed, before a model is asked to
                # narrate anything. Calling the advisor from here skipped
                # scorecard.score entirely, so every pre-registered
                # recommendation stayed open forever and the scorecard the brief
                # renders was permanently empty.
                await loop.weekly(self, guild.id)
            except Exception:
                traceback.print_exc()

    @tasks.loop(hours=24)
    async def daily_alerts(self) -> None:
        """Only speaks when something is actually wrong. No news, no message."""
        for guild in list(self.guilds):
            if not self._rooms.get(guild.id):
                continue
            # First, and in its own try: this is the pass that closes
            # recommendations whose horizon has run out, and it commits before
            # it delivers. Weekly-only grading would leave a 14-day bet unjudged
            # for up to another week. It stays quiet unless a verdict landed or
            # a watched number actually moved.
            try:
                await loop.nightly(self, guild.id)
            except Exception:
                traceback.print_exc()
            # Second, and never able to skip the first. Sharing one try with the
            # grading pass meant a deleted channel or a revoked permission here
            # stopped every recommendation in the server from ever being graded,
            # leaving a stack trace and a comment claiming grading is never
            # skipped.
            try:
                pending = snapshot.unanswered_questions(guild.id, room.owner_id(guild.id) or 0)
                if pending:
                    lines = ["**%d unanswered question(s).**" % len(pending), ""]
                    for q in pending[:5]:
                        lines.append(
                            "%s in #%s, %s days ago: %s\n%s"
                            % (q["author"], q["channel"], q["asked_days_ago"],
                               q["text"][:180], q["link"])
                        )
                    await notify.deliver(
                        self, guild.id, "\n\n".join(lines), "unanswered"
                    )
            except Exception:
                traceback.print_exc()

    @tasks.loop(hours=config.PRESENCE_SAMPLE_HOURS)
    async def hourly_facts(self) -> None:
        """Re-read the server settings, and sample how many people are online.

        The cached Guild never has the approximate counts populated, so the
        fetch is not optional. It costs one request an hour and is the only way
        cadybot learns anything about members who are present and silent —
        without it, "nobody is talking" and "nobody is here" look the same.
        """
        for guild in list(self.guilds):
            try:
                await self._refresh_facts(guild)
                fetched = await self.fetch_guild(guild.id, with_counts=True)
                db.record_presence_sample(
                    guild.id,
                    fetched.approximate_member_count,
                    fetched.approximate_presence_count,
                )
            except discord.HTTPException as exc:
                print("hourly refresh failed for %s: %s" % (guild.id, exc))
            except Exception:
                traceback.print_exc()
            # Its own try, for the reason daily_alerts splits its two: a Discord
            # failure above must not stop the ledger, which touches no network
            # at all. This runs here rather than in a loop of its own because
            # hourly_facts is the one pass that starts under both scheduler
            # modes, so the ledger has exactly one writer on every deployment.
            try:
                ledger.record_day(guild.id, snapshot.build(guild.id))
                ledger.prune(guild.id)
            except Exception:
                traceback.print_exc()

    @tasks.loop(minutes=config.SCAN_INTERVAL_MINUTES)
    async def think_pass(self) -> None:
        """The desk. Wakes on a timer, thinks only when something happened.

        The timer is not the trigger. `agenda.next_provocation` returns None
        unless a row appeared in the database that cadybot has not already
        thought about, so the common outcome here is a handful of indexed
        SELECTs and nothing else — no model call, no message, no cost. On a
        server where nobody is talking that is every tick, indefinitely, which
        is the correct behaviour rather than a degenerate one.

        Like the two reporting loops, it skips guilds with no private channel:
        there is nowhere to say anything, and thinking about a server the
        founder has not set up yet is work nobody asked for.
        """
        for guild in list(self.guilds):
            if not self._rooms.get(guild.id):
                continue
            try:
                await thinking.think(self, guild.id)
            except Exception:
                traceback.print_exc()

    @weekly_brief.before_loop
    async def _wait_weekly(self) -> None:
        await self.wait_until_ready()

    @think_pass.before_loop
    async def _wait_think(self) -> None:
        await self.wait_until_ready()
        # discord.ext.tasks sleeps *after* the body on a relative interval, so
        # an interval loop runs once the moment it starts. With Restart=always
        # and RestartSec=10 a crash loop would therefore attempt a pass every
        # ten seconds, on the tick most likely to find a cold backend. The
        # daily ceiling bounds the damage; this stops it happening at all, and
        # costs a minute of latency on a pass that runs four times a day.
        await asyncio.sleep(config.THINK_START_DELAY_SECONDS)

    @daily_alerts.before_loop
    async def _wait_daily(self) -> None:
        await self.wait_until_ready()

    @hourly_facts.before_loop
    async def _wait_hourly(self) -> None:
        await self.wait_until_ready()


def _why_quiet(row) -> str:
    """One phrase saying what happened to a thought, and why it was not said.

    Silence has several causes now — nothing provoked it, the model declined, a
    figure did not check out, the clock was wrong, the backend may not speak —
    and a founder who cannot tell them apart learns to read all of them as a
    bug.
    """
    if row["outcome"] == "failed":
        return "failed: %s" % (row["failure"] or "unknown")
    if row["surfaced_at"]:
        return "told you"
    unverified = (row["unverified"] or "").strip()
    if unverified not in ("", "[]"):
        return "held back — I could not check %s" % unverified
    if not row["wanted_telling"]:
        return "not worth telling you"
    if config.BACKEND not in config.THINK_SURFACE_BACKENDS:
        return "written, but I do not volunteer on the local model"
    return "written, waiting for a good moment"


def _audit_extra(entry) -> Optional[str]:
    """The numeric part of an audit entry's extras, and nothing else.

    `extra` can carry channel names, automod rule names and moderator-supplied
    reasons. None of that belongs in a database whose rows are fed to a model,
    so only counts survive.
    """
    extra = getattr(entry, "extra", None)
    fields = {}
    for key in ("delete_member_days", "members_removed", "count"):
        value = getattr(extra, key, None)
        if isinstance(value, int):
            fields[key] = value
    return json.dumps(fields) if fields else None


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
            # One lookup round before a verdict. The interaction is already
            # deferred, so the 15-minute window absorbs it comfortably.
            inq = await asyncio.to_thread(
                inquiry.investigate, interaction.guild_id, question,
                config.INQUIRY_ROUNDS_ASK, config.INQUIRY_BUDGET_ASK,
            )
            verdict = await asyncio.to_thread(
                advisor.ask, question, snap, interaction.guild_id, None, inq
            )
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
            result = await asyncio.to_thread(
                advisor.brief, snap, interaction.guild_id, register_now=False
            )
        except (advisor.Refused, advisor.BackendError) as exc:
            await _reply(interaction, str(exc), True)
            return
        await _reply(interaction, advisor.render_brief(result), not visible)
        # After the reply, never before it: the bet is only real once the
        # founder has the recommendation that proposes it.
        advisor.register(result, snap, interaction.guild_id)

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

    @tree.command(name="notes", description="What cadybot thought about on its own")
    @app_commands.guild_only()
    async def notes(interaction: discord.Interaction) -> None:
        """Everything the desk thought, including what it chose not to say.

        The point of showing the unsaid ones is that silence stops being
        ambiguous. Without this, a founder cannot tell "nothing happened" from
        "something broke", and learns to read quiet as a bug.
        """
        await interaction.response.defer(ephemeral=True)
        if not bot.may_read(interaction):
            await _reply(interaction, NO_ROOM, True)
            return
        rows = agenda.recent(interaction.guild_id, limit=8)
        seen = agenda.scans_today(interaction.guild_id)
        looked = ("I have checked **%d times today** and found nothing new. "
                  % seen["scans"] if seen.get("scans") else "")
        if not rows:
            state = thinking.preview(interaction.guild_id)
            await _reply(
                interaction,
                "%sI only spend a model call when something has actually "
                "happened — right now: _%s_." % (looked, state["why"]),
                True,
            )
            return
        lines = ["**What I have been thinking about**", "",
                 looked.strip() or "", ""]
        for row in rows:
            when = row["started_at"][:10]
            lines.append("`%s` **%s**%s — %s" % (
                when, row["kind"],
                " (%s)" % row["about_ref"] if row["about_ref"] else "",
                _why_quiet(row),
            ))
            # The sentence it composed, whether or not it was allowed to say it.
            # Showing only note_to_self meant the two things this command exists
            # for — what it wanted to tell you, and why it did not — were the
            # two things it did not show.
            if row["to_founder"]:
                lines.append("> %s" % row["to_founder"])
            elif row["note_to_self"]:
                lines.append("> _(to myself)_ %s" % row["note_to_self"])
            lines.append("")
        lines.append("_`/quiet 7` stops me volunteering anything for a week._")
        await _reply(interaction, "\n".join(lines), True)

    @tree.command(name="quiet", description="Stop cadybot volunteering thoughts for a while")
    @app_commands.describe(days="How many days of quiet. 0 to turn it back on.")
    @app_commands.guild_only()
    async def quiet(interaction: discord.Interaction, days: int = 7) -> None:
        """Silences speaking, never thinking.

        The journal keeps filling and `/notes` keeps showing it, so turning the
        volume down does not cost the record. Stopping the thinking entirely is
        CADYBOT_THINK_CALLS_PER_DAY=0, which is an operator decision rather than
        a chat one.
        """
        await interaction.response.defer(ephemeral=True)
        if not bot.may_read(interaction):
            await _reply(interaction, NO_ROOM, True)
            return
        days = max(0, min(days, 365))
        if not days:
            db.set_setting(interaction.guild_id, agenda.QUIET_KEY, None)
            await _reply(interaction, "I'll speak up again when something happens.", True)
            return
        until = db.iso(
            datetime.now(timezone.utc) + timedelta(days=days)
        )
        db.set_setting(interaction.guild_id, agenda.QUIET_KEY, until)
        await _reply(
            interaction,
            "Quiet for %d day(s). I'll keep thinking and keep the notes — "
            "`/notes` shows them, and `/quiet 0` turns me back on." % days,
            True,
        )

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
