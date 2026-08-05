# cadybot

A read-only Discord growth advisor.

A caddy knows the course, carries your clubs, and tells you which club to use —
but never takes the swing. cadybot reads your server, keeps a durable record of
it, tells you what to do, and later tells you honestly whether that worked.

It posts in exactly one place: a private channel you create with `/private`.

## The two rules everything else follows from

**1. The model never produces a number.** `snapshot.py` computes every metric in
SQL; the model only interprets what it is handed. That is what stops it
inventing a member count or a retention rate.

**2. The model never grades itself.** `scorecard.py` decides whether past advice
worked, using thresholds fixed *before* the outcome was known, and it cannot
import `advisor` or `llm`. Verdicts reach the prompt as facts the model did not
produce and cannot revise.

The second rule exists because the first one is not enough. A system that cannot
hallucinate a statistic can still hallucinate that it has been helping.

## What it does

- **Listens.** Messages, edits, deletes, joins, leaves, threads, mentions,
  reactions, voice presence counts, invite attribution, moderation actions from
  the audit log, and server settings. Joins and leaves exist only as live
  gateway events — Discord will not let you backfill them — so the listener is
  the one part worth running before anything else is built.
- **Measures.** Stage, activation, response rate, lurker conversion, retention
  brackets, contributor concentration, reply reciprocity, dead channels,
  moderation load. Every rate is gated by sample size.
- **Advises.** `/brief` gives at most three ranked things to do — or zero, with
  a reason citing a number. `/ask` gives a straight yes / no / not-yet.
- **Grades itself.** Each recommendation pre-registers a metric, a direction, a
  horizon and a guardrail *before* you act. Later, the scorer says whether it
  worked, failed, was never attempted, or cannot be told apart from noise.

It never moderates, never DMs members, and never posts outside its own channel.
That last part is enforced in `notify._guard`, which rejects every destination
except the private channel — not by convention.

## What it refuses to tell you

This is a feature, and the thing most tools in this space get wrong.

- **Unknown is not zero.** Every server setting cadybot cannot read comes back as
  `{value: null, reason: ...}`. "Onboarding is off" and "I lack the permission to
  see whether onboarding is off" are different sentences.
- **Small samples get no percentage.** A rate over seven people looks exactly as
  precise as a rate over seven thousand. Below the floors in `stats.py`, cadybot
  prints the counts and withholds the ratio.
- **No verdict from noise.** Below 20 baseline events, the scorer returns
  `inconclusive` rather than a confident call. On a seven-member server this
  makes `worked` and `harmful` equally unreachable — deliberately symmetric.
- **`not_measurable`** names what no bot can see at any permission level, so the
  gap is visible rather than quietly filled in. Discord's own visitor and
  retention numbers are dashboard-only; the `VIEW_GUILD_INSIGHTS` permission bit
  exists but no REST route consumes it.

## Talking to it

Inside the private channel, just type. Any ordinary message is a question, with
the same stage gates and evidence discipline as `/brief` — it will still tell
you no. It remembers the last several turns, and that survives a restart.

Start a line with `//` to say something it should ignore. `/reset` clears the
thread when you change subject.

## Commands

`/private`, `/add`, `/remove` and `/backfill` need **Manage Server**; the rest
need to be able to see the private channel.

```
/private            create (or repair) your private channel here
/ask <question>     a straight yes / no / not-yet
/brief              what to do this week, ranked — possibly nothing
/snapshot           the raw numbers, no interpretation
/who                who can see the private channel, including anyone
                    bypassing it via Administrator
/add /remove        change that
/backfill           import this server's message history
/reset              forget the conversation so far
```

Answers are visible in the private channel and ephemeral anywhere else, so
cadybot never puts analytics in front of your members.

**`/who` tells the truth about privacy.** Administrators bypass channel
overwrites entirely, so a server admin — or an admin *bot* — can read the
private channel and `/remove` cannot stop them. Only removing Administrator
from their role can.

## Several servers at once

Each server is independent: its own private channel, owner, numbers and
recommendations. Every row is keyed by `guild_id` and no query crosses servers,
so a test server and a live server can never see each other. Run `/private`
separately in each; whoever runs it becomes that server's owner.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env          # add DISCORD_TOKEN
python -m cadybot invite      # prints your OAuth URLs
python -m cadybot listen
```

Enable **Server Members Intent** and **Message Content Intent** under
Developer Portal → Bot → Privileged Gateway Intents. Nothing works without both.
Under 10,000 reachable users they are plain toggles.

The invite URL includes `applications.commands`; without it the slash commands
never appear.

**Hosting:** `deploy/install.sh` for a Mac (launchd), `deploy/linux/` for a VM
(systemd). See `deploy/linux/README.md` — the short version is that the bot
belongs on the smallest VM you can rent and the model does not belong on a VM at
all.

## Model backend

`CADYBOT_BACKEND=anthropic` uses Claude Opus 5. `CADYBOT_BACKEND=ollama` runs
locally and free.

Local inference is for offline development, not for saving money. The stage
gates and playbooks are prompt-level, so a weaker model follows them less
reliably — and judgment is the entire product. At this volume the API costs a
few dollars a month.

`CADYBOT_OLLAMA_KEEP_ALIVE=0` unloads the model between questions instead of
holding several gigabytes; `30m` keeps it warm at the cost of that memory.
`CADYBOT_OLLAMA_NUM_CTX` must be large enough for the whole prompt — a truncated
stage gate is worse than none, so cadybot refuses a response whose prompt nearly
filled the window rather than trusting it.

## Tests

```bash
.venv/bin/python tests/harness.py            # 19 synthetic servers, no model
.venv/bin/python tests/scorer.py             # 31 grading cases, no model
.venv/bin/python tests/harness.py --advice    # runs the model, slow
.venv/bin/python tests/conversation.py        # multi-turn dialogue, runs the model
```

The first two are deterministic and fast — run them after any change. They cover
brand-new servers, raids, exoduses, all-bot servers, one power user carrying
everything, 5,000 members, dead channels, moderation waves, and the difference
between a fact that is false and a fact cadybot cannot read.

The scorer suite is the one that matters most. It pins the cases where an
earlier version marked its own advice `worked` while twenty people went silent.

## Data

Everything lands in `cadybot.db` (gitignored). `python -m cadybot purge --guild
<id> --member <id>` deletes one member's data on request.

Messages in a private channel are never stored, and bot messages are excluded
from activity counts, so cadybot cannot inflate its own numbers.

## Compliance notes

- Discord's Developer Policy prohibits using API message content to **train**
  ML/AI models. cadybot only ever runs **inference** — no fine-tuning, ever.
  Local inference sends message content nowhere; Anthropic does not train on API
  inputs.
- Developer Terms allow retaining chat logs only as necessary for operation, and
  require deleting end-user data on request. Hence `purge`.
- Message content is never written to logs — only IDs and counts.
- Put one line in `#rules` telling members the server is analyzed by a private
  bot. Output invisibility is fine; existence invisibility is not.

Read §21 of the Developer Policy yourself before making this public.
