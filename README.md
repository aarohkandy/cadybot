# cadybot

A read-only growth advisor for the Cady Discord server.

A caddy knows the course, carries your clubs, and tells you which one to use —
but never takes the swing. cadybot reads your server, keeps a durable record of
it, and tells you what to do. It never posts in the server.

## What it does

- **Listens.** Logs messages, joins, leaves, and voice-channel *presence counts*
  to SQLite. Joins and leaves cannot be backfilled from Discord — they exist only
  as live gateway events — so the listener is the only thing that matters on day one.
- **Catches.** Flags unanswered questions from members and members who have gone
  quiet. At seven members, one ignored message is a churned user.
- **Advises.** `ask` gives you a straight yes / no / not-yet on a specific idea,
  grounded in the actual numbers. `brief` gives you at most three ranked things
  to do, each with evidence.
- **Remembers.** Every recommendation is stored so it can tell you later whether
  its own advice worked.

It does not post, react, DM members, or moderate. Output goes to your DMs and
(optionally) one locked staff channel.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Create a Discord application at <https://discord.com/developers/applications>,
add a bot, and under **Bot → Privileged Gateway Intents** enable:

- **Server Members Intent**
- **Message Content Intent**

Under 10,000 reachable users these are toggles, no application needed.

Invite it with `bot` scope and read-only permissions: View Channels,
Read Message History. Nothing that can write.

Copy `.env.example` to `.env` and fill it in.

## Use

```bash
python -m cadybot listen                      # run the listener (do this first, keep it up)
python -m cadybot backfill                    # one-time history import
python -m cadybot snapshot                    # print the raw numbers, no LLM
python -m cadybot ask "would a weekly event help?"
python -m cadybot brief
python -m cadybot outcomes                    # did past advice work?
```

Once `listen` is running you can also DM the bot directly:

```
ask would a weekly event help?
brief
snapshot
```

## Data

Everything lands in `cadybot.db` (gitignored). Message text is kept in full for
now — seven people cannot generate enough to matter. Revisit a rolling window
at ~1,000 members. `python -m cadybot purge --member <id>` deletes one member's
data on request.

Every table carries `guild_id` and no query crosses guilds, so this can go
multi-tenant later without a schema rewrite.

## Compliance notes

- Discord's Developer Policy prohibits using message content obtained via the
  API to **train** ML/AI models. This sends message text to Claude for
  **inference** only — no fine-tuning, ever. Anthropic does not train on API
  inputs.
- Discord's Developer Terms allow retaining chat logs only as necessary for
  operation, and require deleting end-user data on request. Hence `purge`.
- Message content is never written to logs — only IDs and counts.
- Add one line to `#rules` telling members the server is analyzed by a private
  bot. Output invisibility is fine; existence invisibility is not.

Read §21 of the Developer Policy yourself before making this public.
