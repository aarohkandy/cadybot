# cadybot

A read-only growth advisor for the Cady Discord server.

A caddy knows the course, carries your clubs, and tells you which club to use —
but never takes the swing. cadybot reads your server, keeps a durable record of
it, and tells you what to do. It posts in exactly one place: a private channel
it creates for you.

## What it does

- **Listens.** Logs messages, joins, leaves, and voice-channel *presence counts*
  to SQLite. Joins and leaves cannot be backfilled from Discord — they exist only
  as live gateway events — so the listener is the only thing that matters on day one.
- **Catches.** Flags unanswered questions from members and members who have gone
  quiet. At seven members, one ignored message is a churned user.
- **Advises.** `ask` gives a straight yes / no / not-yet on a specific idea,
  grounded in the actual numbers. `brief` gives at most three ranked things to
  do, each with evidence and a metric to watch.
- **Remembers.** Every recommendation is stored so it can later tell you whether
  its own advice worked.

It never moderates, never DMs members, and never posts in any channel except its
own. That last part is enforced in code, not by convention: every outgoing
message passes a guard that rejects any destination other than your DMs and the
private channel — see `notify.WriteBlocked`.

## The private channel

On startup cadybot creates a channel (default `#cadybot`), hidden from
`@everyone`, containing you and it. That is where briefs land and where you talk
to it.

```
who                 who can currently see it
add @someone        let someone in
remove @someone     take them back out
```

Anyone in the channel can run `ask`, `brief`, `snapshot` and `who`.
`add`, `remove` and `backfill` are yours only. Ordinary conversation in there is
ignored, so it works as a real chat rather than a command prompt.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env
```

Create an application at <https://discord.com/developers/applications>, add a
bot, copy the token into `.env`, then under **Bot → Privileged Gateway Intents**
enable **Server Members Intent** and **Message Content Intent**. Under 10,000
reachable users these are plain toggles — no application to file.

Then print your invite link:

```bash
python -m cadybot invite
```

It prints two URLs: a scoped one with exactly what cadybot uses, and an
Administrator one for a throwaway test server. The scoped set is View Channels,
Send Messages, Read Message History, Manage Channels, Manage Roles, Manage
Server. The two Manage permissions are what let it create the private channel
and control who sees it; Manage Server is what lets it read invite use-counts to
attribute joins.

## Use

```bash
python -m cadybot doctor                      # is the model backend reachable?
python -m cadybot listen --backfill           # start logging + import history
python -m cadybot snapshot                    # raw numbers, no model involved
python -m cadybot ask "would a weekly event help?"
python -m cadybot brief
python -m cadybot outcomes                    # did past advice work?
```

Keep `listen` running. Everything else also works from inside the private
channel, which is the intended way to use it day to day.

## Model backend

`CADYBOT_BACKEND=ollama` (default) runs inference locally and free.
`CADYBOT_BACKEND=anthropic` uses Claude Opus 5.

The local path is for exercising the plumbing. A 9GB local model will follow the
stage gates and cite the snapshot, but its judgment is visibly thinner than the
API's, and judgment is the entire product. Treat local output as proof the wiring
works, not as advice worth acting on.

`CADYBOT_OLLAMA_NUM_CTX` matters more than the model choice. Ollama's default
context is small enough to silently truncate the system prompt, and a truncated
stage gate is worse than none — the model would happily recommend a tournament
to seven people. cadybot refuses to trust a response whose prompt nearly filled
the window.

## Data

Everything lands in `cadybot.db` (gitignored). Message text is kept in full for
now — seven people cannot generate enough to matter. Revisit a rolling window at
~1,000 members. `python -m cadybot purge --member <id>` deletes one member's
data on request.

Messages in the private channel are never stored, so cadybot's own output can
never inflate its own numbers. Bot messages are excluded from activity counts
for the same reason: a chatty bot is not a lively server.

Every table carries `guild_id` and no query crosses guilds, so this can go
multi-tenant later without a schema rewrite.

## Compliance notes

- Discord's Developer Policy prohibits using message content obtained via the
  API to **train** ML/AI models. This sends message text to a model for
  **inference** only — no fine-tuning, ever. Local inference sends it nowhere at
  all; Anthropic does not train on API inputs.
- Discord's Developer Terms allow retaining chat logs only as necessary for
  operation, and require deleting end-user data on request. Hence `purge`.
- Message content is never written to logs — only IDs and counts.
- Add one line to `#rules` telling members the server is analyzed by a private
  bot. Output invisibility is fine; existence invisibility is not.

Read §21 of the Developer Policy yourself before making this public.
