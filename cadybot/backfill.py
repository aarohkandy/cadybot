"""One-time history import.

Discord serves at most 100 messages per request; discord.py paginates and honours
rate limits for us. The sleep between channels is politeness, not necessity.

Threads are separate channels and must be enumerated explicitly — both active and
archived. Forum channels contain nothing but threads, so skipping this step loses
those entirely.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import discord

from . import db

TEXTLIKE = (discord.TextChannel, discord.Thread, discord.VoiceChannel, discord.StageChannel)


def _kind(channel) -> str:
    return type(channel).__name__


def register_channel(guild_id: int, channel) -> None:
    """Write one channel or thread row. Shared with the live listener.

    Guild.channels returns only GuildChannel, so threads reach the database by a
    different route and had better be described the same way when they do. A
    thread's parent is a channel; an ordinary channel's parent is its category.
    """
    is_thread = isinstance(channel, discord.Thread)
    parent = channel.parent if is_thread else getattr(channel, "category", None)
    db.upsert_channel(
        guild_id,
        channel.id,
        getattr(channel, "name", None),
        _kind(channel),
        getattr(parent, "id", None),
        db.iso(getattr(channel, "created_at", None)),
        archived=int(bool(channel.archived)) if is_thread else None,
        archive_timestamp=db.iso(channel.archive_timestamp) if is_thread else None,
        auto_archive_duration=channel.auto_archive_duration if is_thread else None,
        thread_message_count=channel.message_count if is_thread else None,
        parent_kind=_kind(parent) if (is_thread and parent is not None) else None,
    )


def reference_of(message: discord.Message) -> Tuple[Optional[int], Optional[int]]:
    """Split the one `reference` field Discord overloads three ways.

    A genuine reply, a forward, and the content-free mirror Discord posts when a
    thread is created all populate message.reference. Treating all three as
    replies made 12 of 17 stored replies phantoms on the live server. A forward
    additionally points at a message in a server cadybot is not in, so its target
    id means nothing here and only the fact of the forward is worth keeping.
    """
    ref = message.reference
    if ref is None:
        return None, None
    if ref.type is discord.MessageReferenceType.forward:
        return None, 1
    if (
        message.type is discord.MessageType.reply
        and ref.type is discord.MessageReferenceType.default
    ):
        return ref.message_id, 0
    return None, None


def message_row(
    guild_id: int, channel_id: int, message: discord.Message, reactions: int
) -> Dict[str, Any]:
    """Build the messages row for one Discord message.

    Shared by history import and live ingest so the two cannot disagree about
    what a reply is. A forward's text lives in message.message_snapshots and was
    written in a server cadybot was never in; it is deliberately not read here.
    """
    reply_to_id, ref_type = reference_of(message)
    return {
        "guild_id": guild_id,
        "channel_id": channel_id,
        "message_id": message.id,
        "author_id": message.author.id,
        "created_at": db.iso(message.created_at),
        "content": message.content or None,
        "reply_to_id": reply_to_id,
        "attachments": len(message.attachments),
        "reactions": reactions,
        "type": message.type.value,
        "ref_type": ref_type,
        "flags": message.flags.value,
        "edited_at": db.iso(message.edited_at),
        "mention_everyone": 1 if message.mention_everyone else 0,
    }


async def _store_channel(guild_id: int, channel) -> int:
    register_channel(guild_id, channel)

    count = 0
    try:
        async for message in channel.history(limit=None, oldest_first=True):
            # History cannot say who reacted, only how many did, so the
            # aggregate goes straight onto the message. Inventing per-user
            # reaction rows to match would be inventing data.
            db.upsert_message(
                message_row(
                    guild_id, channel.id, message, sum(r.count for r in message.reactions)
                )
            )
            db.add_mentions(
                guild_id,
                message.id,
                message.author.id,
                [u.id for u in message.mentions],
                db.iso(message.created_at),
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


async def run(guild: discord.Guild, skip_channel_id: Optional[int] = None) -> int:
    total = 0

    for member in guild.members:
        db.upsert_member(
            guild.id,
            member.id,
            member.name,
            member.display_name,
            member.bot,
            db.iso(member.joined_at),
            db.member_state(member),
        )

    for channel in guild.channels:
        if skip_channel_id is not None and channel.id == skip_channel_id:
            continue  # cadybot's own console is not server activity
        if not isinstance(channel, TEXTLIKE):
            register_channel(guild.id, channel)
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
