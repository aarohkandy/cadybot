"""Delivery. The only module allowed to send anything.

Every send passes `_guard`, which permits exactly two destinations: a DM, and
the private channel cadybot created. Every other channel in the server raises.
That is what makes "never posts in the server" a property of the code rather
than a thing we remembered to do.
"""

from typing import List, Optional

import discord

from . import config, room

LIMIT = 1900  # Discord's cap is 2000; leave room for fence characters


class WriteBlocked(RuntimeError):
    pass


def _guard(destination) -> None:
    if isinstance(destination, discord.DMChannel):
        return  # DMs only ever go to a server's registered owner

    guild = getattr(destination, "guild", None)
    if guild is not None:
        allowed = room.stored_id(guild.id)
        if allowed and getattr(destination, "id", None) == allowed:
            return

    raise WriteBlocked(
        "cadybot is read-only outside its own channel. Refusing to write to %r."
        % getattr(destination, "name", destination)
    )


def chunk(text: str, limit: int = LIMIT) -> List[str]:
    """Split on paragraph, then line, then word. Never mid-word if avoidable."""
    out: List[str] = []
    remaining = text.strip()
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n\n")
        if cut < limit // 3:
            cut = window.rfind("\n")
        if cut < limit // 3:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit
        out.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        out.append(remaining)
    return out


async def send(destination, text: str) -> None:
    _guard(destination)
    for part in chunk(text):
        await destination.send(part)


async def deliver(bot: discord.Client, guild_id: int, text: str) -> None:
    """Post to that server's private channel; fall back to DMing its owner.

    Scoped to one server on purpose — a brief about the test server must never
    land in the live server's channel.
    """
    guild = bot.get_guild(guild_id)
    if guild is not None:
        channel_id = room.stored_id(guild_id)
        channel = guild.get_channel(channel_id) if channel_id else None
        if channel is not None:
            await send(channel, text)
            return

    owner = room.owner_id(guild_id)
    if not owner:
        return
    user = bot.get_user(owner) or await bot.fetch_user(owner)
    if user is not None:
        await send(await user.create_dm(), text)
