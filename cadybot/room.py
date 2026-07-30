"""The private channel.

cadybot creates one channel, hidden from @everyone, containing you and it. That
channel is where you talk to it and where briefs land, and you control its
membership with `add` / `remove`.

This is the one place cadybot writes in the server. Everywhere else it is still
read-only — see notify.WriteBlocked, which enforces that structurally rather
than by convention.
"""

from typing import List, Optional

import discord

from . import config, db

SETTING = "room_channel_id"

TOPIC = "Private. cadybot posts here and nowhere else. `help` for commands."

WELCOME = (
    "**cadybot is listening.**\n\n"
    "This channel is private — only the people listed here can see it. "
    "Use `add @someone` / `remove @someone` to change that.\n\n"
    "`ask <question>` — a straight yes / no / not-yet\n"
    "`brief` — what to do this week\n"
    "`snapshot` — the raw numbers, no interpretation\n"
    "`who` — who can see this channel\n"
    "`backfill` — re-import history\n\n"
    "It reads every other channel and never writes to them."
)


class RoomError(RuntimeError):
    pass


def _overwrites(guild: discord.Guild, owner: discord.Member) -> dict:
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
    raw = db.get_setting(guild_id, SETTING)
    return int(raw) if raw else None


async def ensure(guild: discord.Guild) -> discord.TextChannel:
    """Find the private channel, or create it. Idempotent."""
    channel = None

    known = stored_id(guild.id)
    if known:
        channel = guild.get_channel(known)

    if channel is None:
        for existing in guild.text_channels:
            if existing.name == config.ROOM_NAME:
                channel = existing
                break

    owner = guild.get_member(config.OWNER_ID)
    if owner is None:
        try:
            owner = await guild.fetch_member(config.OWNER_ID)
        except discord.NotFound:
            raise RoomError(
                "OWNER_ID %s is not a member of %s." % (config.OWNER_ID, guild.name)
            )

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
                "cadybot needs Manage Channels and Manage Roles to create its "
                "private channel. Re-invite it with the permissions in the README."
            )
        db.set_setting(guild.id, SETTING, str(channel.id))
        await channel.send(WELCOME)
        return channel

    # Channel already exists — make sure it is actually private and that both
    # cadybot and the owner can use it. Cheap to re-assert, and it repairs a
    # channel someone edited by hand.
    db.set_setting(guild.id, SETTING, str(channel.id))
    try:
        for target, overwrite in _overwrites(guild, owner).items():
            if channel.overwrites_for(target) != overwrite:
                await channel.set_permissions(target, overwrite=overwrite)
    except discord.Forbidden:
        pass  # usable as-is; just can't self-repair

    return channel


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
        raise RoomError("cadybot needs Manage Roles to change who can see this channel.")


async def remove(channel: discord.TextChannel, member: discord.Member) -> None:
    if member.id == config.OWNER_ID:
        raise RoomError("You can't remove yourself.")
    if member.id == channel.guild.me.id:
        raise RoomError("Removing cadybot from its own channel would be unwise.")
    try:
        await channel.set_permissions(member, overwrite=None, reason="removed from cadybot channel")
    except discord.Forbidden:
        raise RoomError("cadybot needs Manage Roles to change who can see this channel.")


def roster(channel: discord.TextChannel) -> List[str]:
    names = []
    for target, overwrite in channel.overwrites.items():
        if isinstance(target, discord.Member) and overwrite.view_channel:
            names.append(target.display_name)
    return sorted(names)
