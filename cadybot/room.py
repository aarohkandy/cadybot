"""The private channel — one per server.

`/private` creates a channel hidden from @everyone containing you and cadybot.
Whoever runs it becomes that server's owner for cadybot's purposes: their
posting cadence is what gets tracked, and their messages are not counted as
unanswered questions.

Each server is fully independent. A test server and a live server get separate
channels, separate owners, separate data, and separate recommendations — the
only thing they share is the process and the SQLite file, and every table is
keyed by guild_id.

This is the one place cadybot writes in a server. Everywhere else is read-only,
enforced in notify._guard rather than by convention.
"""

from typing import Dict, List, Optional

import discord

from . import config, db

CHANNEL_KEY = "room_channel_id"
OWNER_KEY = "owner_id"

TOPIC = "Private. cadybot posts here and nowhere else."

WELCOME = (
    "**cadybot is listening.**\n\n"
    "This channel is private — only the people listed here can see it.\n\n"
    "`/ask <question>` — a straight yes / no / not-yet\n"
    "`/brief` — what to do this week\n"
    "`/snapshot` — the raw numbers, no interpretation\n"
    "`/who` — who can see this channel\n"
    "`/add` and `/remove` — change that\n"
    "`/backfill` — import message history\n\n"
    "It reads every other channel in this server and never writes to them. "
    "Other servers cadybot is in are kept entirely separate from this one.\n\n"
    "Nothing said in here is counted as server activity."
)


class RoomError(RuntimeError):
    pass


def _overwrites(guild: discord.Guild, owner: discord.Member) -> Dict[object, discord.PermissionOverwrite]:
    return {
        guild.default_role: discord.PermissionOverwrite(view_channel=False),
        guild.me: discord.PermissionOverwrite(
            view_channel=True,
            send_messages=True,
            read_message_history=True,
            manage_channels=True,
            embed_links=True,
        ),
        owner: discord.PermissionOverwrite(
            view_channel=True, send_messages=True, read_message_history=True
        ),
    }


def stored_id(guild_id: int) -> Optional[int]:
    raw = db.get_setting(guild_id, CHANNEL_KEY)
    return int(raw) if raw else None


def owner_id(guild_id: int) -> Optional[int]:
    raw = db.get_setting(guild_id, OWNER_KEY)
    if raw:
        return int(raw)
    return config.OWNER_ID


async def create(guild: discord.Guild, owner: discord.Member) -> discord.TextChannel:
    """Create the private channel, or repair and re-hand-over an existing one.

    Idempotent: running /private again re-asserts the permissions, which also
    fixes a channel somebody edited by hand.
    """
    channel = None

    known = stored_id(guild.id)
    if known:
        channel = guild.get_channel(known)
    if channel is None:
        for existing in guild.text_channels:
            if existing.name == config.ROOM_NAME:
                channel = existing
                break

    if channel is None:
        try:
            channel = await guild.create_text_channel(
                config.ROOM_NAME,
                overwrites=_overwrites(guild, owner),
                topic=TOPIC,
                reason="cadybot private channel",
            )
        except discord.Forbidden:
            raise RoomError(
                "I need **Manage Channels** and **Manage Roles** to create the channel. "
                "Check my role in Server Settings → Roles, or re-invite me with "
                "`python -m cadybot invite`."
            )
        _remember(guild.id, channel.id, owner.id)
        await channel.send(WELCOME)
        return channel

    try:
        for target, overwrite in _overwrites(guild, owner).items():
            if channel.overwrites_for(target) != overwrite:
                await channel.set_permissions(target, overwrite=overwrite)
    except discord.Forbidden:
        raise RoomError(
            "#%s already exists but I can't manage its permissions. Give me "
            "**Manage Roles**, or delete the channel and run `/private` again." % channel.name
        )

    _remember(guild.id, channel.id, owner.id)
    return channel


def _remember(guild_id: int, channel_id: int, owner: int) -> None:
    db.set_setting(guild_id, CHANNEL_KEY, str(channel_id))
    db.set_setting(guild_id, OWNER_KEY, str(owner))


async def add(channel: discord.TextChannel, member: discord.Member) -> None:
    try:
        await channel.set_permissions(
            member,
            overwrite=discord.PermissionOverwrite(
                view_channel=True, send_messages=True, read_message_history=True
            ),
            reason="added to cadybot channel",
        )
    except discord.Forbidden:
        raise RoomError("I need **Manage Roles** to change who can see this channel.")


async def remove(channel: discord.TextChannel, member: discord.Member) -> None:
    if member.id == owner_id(channel.guild.id):
        raise RoomError("That's the owner of this channel — removing them would lock it.")
    if member.id == channel.guild.me.id:
        raise RoomError("Removing cadybot from its own channel would be unwise.")
    try:
        await channel.set_permissions(member, overwrite=None, reason="removed from cadybot channel")
    except discord.Forbidden:
        raise RoomError("I need **Manage Roles** to change who can see this channel.")


def roster(channel: discord.TextChannel) -> List[str]:
    names = []
    for target, overwrite in channel.overwrites.items():
        if isinstance(target, discord.Member) and overwrite.view_channel:
            names.append(target.display_name)
    return sorted(names)


def forget(guild_id: int) -> None:
    """Called when the channel is deleted or cadybot is kicked."""
    db.set_setting(guild_id, CHANNEL_KEY, None)
