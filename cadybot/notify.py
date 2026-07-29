"""Delivery. The only module allowed to send anything.

Every send goes through `_guard`, which permits exactly two destinations: a DM to
the owner, and the one configured staff channel. Nothing else can be written to,
which is what makes "never posts in the server" a structural property rather
than a promise.
"""

from typing import Iterable, List, Optional

import discord

from . import config

LIMIT = 1900  # Discord's cap is 2000; leave room for fence characters


class WriteBlocked(RuntimeError):
    pass


def _guard(destination) -> None:
    if isinstance(destination, discord.DMChannel):
        recipient = getattr(destination, "recipient", None)
        if recipient is None or recipient.id == config.OWNER_ID:
            return
        raise WriteBlocked("cadybot may only DM the owner.")
    if config.STAFF_CHANNEL_ID and getattr(destination, "id", None) == config.STAFF_CHANNEL_ID:
        return
    raise WriteBlocked(
        "cadybot is read-only in servers. Refusing to write to %r." % destination
    )


def chunk(text: str, limit: int = LIMIT) -> List[str]:
    """Split on paragraph, then line, then hard-cut. Never mid-word if avoidable."""
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


async def deliver(bot: discord.Client, text: str, also_staff: bool = True) -> None:
    """DM the owner, and mirror to the staff channel if one is configured."""
    targets: List[object] = []

    owner = bot.get_user(config.OWNER_ID) if config.OWNER_ID else None
    if owner is None and config.OWNER_ID:
        owner = await bot.fetch_user(config.OWNER_ID)
    if owner is not None:
        targets.append(await owner.create_dm())

    if also_staff and config.STAFF_CHANNEL_ID:
        channel = bot.get_channel(config.STAFF_CHANNEL_ID)
        if channel is not None:
            targets.append(channel)

    for target in targets:
        await send(target, text)
