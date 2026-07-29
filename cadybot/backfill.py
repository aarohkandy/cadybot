"""One-time history import.

Discord serves at most 100 messages per request; discord.py paginates and honours
rate limits for us. The sleep between channels is politeness, not necessity.

Threads are separate channels and must be enumerated explicitly — both active and
archived. Forum channels contain nothing but threads, so skipping this step loses
those entirely.
"""

import asyncio
from typing import List, Optional

import discord

from . import db

TEXTLIKE = (discord.TextChannel, discord.Thread, discord.VoiceChannel, discord.StageChannel)


def _kind(channel) -> str:
    return type(channel).__name__


async def _store_channel(guild_id: int, channel) -> int:
    db.upsert_channel(
        guild_id,
        channel.id,
        getattr(channel, "name", None),
        _kind(channel),
        getattr(getattr(channel, "parent", None), "id", None),
        db.iso(getattr(channel, "created_at", None)),
    )

    count = 0
    try:
        async for message in channel.history(limit=None, oldest_first=True):
            db.upsert_message(
                {
                    "guild_id": guild_id,
                    "channel_id": channel.id,
                    "message_id": message.id,
                    "author_id": message.author.id,
                    "created_at": db.iso(message.created_at),
                    "content": message.content or None,
                    "reply_to_id": (
                        message.reference.message_id if message.reference else None
                    ),
                    "attachments": len(message.attachments),
                    "reactions": sum(r.count for r in message.reactions),
                }
            )
            count += 1
    except discord.Forbidden:
        print("  skipped #%s (no read access)" % getattr(channel, "name", channel.id))
    except discord.HTTPException as exc:
        print("  #%s failed: %s" % (getattr(channel, "name", channel.id), exc))
    return count


async def _threads(channel) -> List[discord.Thread]:
    found: List[discord.Thread] = list(getattr(channel, "threads", []) or [])
    archived = getattr(channel, "archived_threads", None)
    if archived is None:
        return found
    try:
        async for thread in archived(limit=None):
            found.append(thread)
    except (discord.Forbidden, discord.HTTPException):
        pass
    return found


async def run(guild: discord.Guild) -> int:
    total = 0

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
        if not isinstance(channel, TEXTLIKE):
            db.upsert_channel(
                guild.id,
                channel.id,
                getattr(channel, "name", None),
                _kind(channel),
                getattr(getattr(channel, "category", None), "id", None),
                db.iso(getattr(channel, "created_at", None)),
            )
            continue

        n = await _store_channel(guild.id, channel)
        total += n
        print("  #%-24s %5d messages" % (getattr(channel, "name", channel.id), n))
        await asyncio.sleep(0.4)

        for thread in await _threads(channel):
            tn = await _store_channel(guild.id, thread)
            total += tn
            print("    -> %-20s %5d messages" % (thread.name[:20], tn))
            await asyncio.sleep(0.4)

    # Forum channels are all threads; catch any not reached above.
    for forum in getattr(guild, "forums", []) or []:
        for thread in await _threads(forum):
            tn = await _store_channel(guild.id, thread)
            total += tn
            print("    -> %-20s %5d messages" % (thread.name[:20], tn))
            await asyncio.sleep(0.4)

    return total
