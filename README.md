# cadybot

A read-only growth advisor for a Discord server.

A caddy knows the course, carries your clubs, and tells you which club to use —
but never takes the swing. cadybot reads your server, keeps a durable record of
it, and tells you what to do. It posts in exactly one place: a private channel
you create with `/private`.

## What it does

- **Listens.** Logs messages, joins, leaves, and voice-channel *presence counts*
  to SQLite. Joins and leaves cannot be backfilled from Discord — they exist only
  as live gateway events — so the listener is the only thing that matters on day one.
- **Catches.** Flags unanswered questions from members and members who have gone
  quiet. At seven members, one ignored message is a churned user.
- **Advises.** `/ask` gives a straight yes / no / not-yet on a specific idea,
  grounded in the actual numbers. `/brief` gives at most three ranked things to
  do, each with evidence and a metric to watch.
- **Remembers.** Every recommendation is stored so it can later tell you whether
  its own advice worked.

It never moderates, never DMs members, and never posts in any channel except its
own. That last part is enforced in code: every outgoing message passes a guard
that rejects any destination other than that server's private channel — see
`notify.WriteBlocked`.

## Several servers at once

Add cadybot to as many servers as you like. Each one is completely independent:
its own private channel, its own owner, its own numbers, its own
recommendations. Every row in the database is keyed by `guild_id` and no query
crosses servers, so a test server and a live server can never see each other.

Run `/private` separately in each. Whoever runs it becomes that server's owner —
the person whose posting cadence is tracked, and whose messages aren't counted
as unanswered questions.

## Talking to it

Inside the private channel, just type. Any ordinary message is a question, and
cadybot answers with the same stage gates and evidence discipline as `/brief` —
it will still tell you no. It remembers the last several turns, so follow-ups
work, and that history survives a restart.

Start a line with `//` to say something it should ignore, so the channel stays
usable for notes and side chat. `/reset` clears the thread when you change
subject.

Commands are still there for the crisp version: `/ask` forces a yes/no/not-yet
verdict, `/brief` produces the ranked list.

## Commands

Run these in Discord. `/private`, `/add`, `/remove` and `/backfill` need
**Manage Server**; the rest need to be able to see the private channel.

```
/private            create (or repair) your private channel here
/ask <question>     a straight yes / no / not-yet
/brief              what to do this week, ranked
/snapshot           the raw numbers, no interpretation
/who                who can see the private channel
/add @someone       let someone in
/remove @someone    take them back out
/backfill           import this server's message history
```

Answers are visible in the private channel and ephemeral (only you see them)
anywhere else, so cadybot never puts analytics in front of your members.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env
```

Create an application at <https://discord.com/developers/applications>, add a
bot, and put the token in `.env`. Under **Bot → Privileged Gateway Intents**
enable **Server Members Intent** and **Message Content Intent** — nothing works
without both. Under 10,000 reachable users these are plain toggles, no
application to file.

```bash
python -m cadybot invite     # prints your OAuth URLs
python -m cadybot listen     # keep this running
```

The invite URL includes the `applications.commands` scope; without it the slash
commands never appear. The scoped permission set is View Channels, Send
Messages, Read Message History, Manage Channels, Manage Roles, Manage Server.
The two Manage permissions let cadybot create the private channel and control
who sees it; Manage Server lets it read invite use-counts to attribute joins.

Slash commands are synced per-server the moment cadybot joins, so they appear
immediately.

## Hosting it on a Mac

```bash
deploy/install.sh              # start now, and at every login
deploy/install.sh --awake      # ...and stop the Mac idle-sleeping while it runs
deploy/install.sh status
deploy/install.sh restart      # after changing code or .env
deploy/install.sh stop
deploy/install.sh uninstall
```

This installs a launchd agent that starts cadybot at login and restarts it if it
dies. Logs go to `logs/cadybot.log` and `logs/cadybot.err`.

**Don't keep the project in `~/Downloads`, `~/Desktop`, or `~/Documents.`**
macOS blocks background agents from reading those directories, and the failure
is an opaque `PermissionError: Operation not permitted` on `.venv/pyvenv.cfg`
rather than anything that mentions permissions. `~/cadybot` is fine.

**While the Mac sleeps, cadybot is offline**, and gateway events during that
window are gone for good. Messages can be recovered later with `/backfill`;
joins and leaves cannot — Discord has no API for them. `--awake` wraps the agent
in `caffeinate -i` so the machine won't idle-sleep while it's running, at a real
cost in battery. Closing the lid still sleeps regardless.

Local inference also needs Ollama running. The Ollama app installs itself as a
background service, so it comes back on its own after a reboot; `python -m
cadybot doctor` tells you if it hasn't.

## Command line

```bash
python -m cadybot doctor                  # is the model backend reachable?
python -m cadybot servers                 # which servers cadybot knows
python -m cadybot snapshot --guild <id>   # raw numbers, no model involved
python -m cadybot ask --guild <id> "would a weekly event help?"
python -m cadybot brief --guild <id>
python -m cadybot outcomes --guild <id>   # did past advice work?
```

`--guild` can be omitted if `GUILD_ID` is set in `.env`, or if cadybot only
knows about one server. With several known and no default it refuses to guess.

## Model backend

`CADYBOT_BACKEND=ollama` (default) runs inference locally and free.
`CADYBOT_BACKEND=anthropic` uses Claude Opus 5.

The local path is for exercising the plumbing. A 9GB local model will follow the
stage gates and cite the snapshot, but its judgment is visibly thinner than the
API's, and judgment is the entire product. Treat local output as proof the
wiring works, not as advice worth acting on.

`CADYBOT_OLLAMA_NUM_CTX` matters more than the model choice. Ollama's default
context is small enough to silently truncate the system prompt, and a truncated
stage gate is worse than none — the model would happily recommend a tournament
to seven people. cadybot refuses to trust a response whose prompt nearly filled
the window.

## Data

Everything lands in `cadybot.db` (gitignored). Message text is kept in full for
now — seven people cannot generate enough to matter. Revisit a rolling window at
~1,000 members. `python -m cadybot purge --guild <id> --member <id>` deletes one
member's data on request.

Messages in a private channel are never stored, so cadybot's own output can
never inflate its own numbers. Bot messages are excluded from activity counts
for the same reason: a chatty bot is not a lively server.

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
