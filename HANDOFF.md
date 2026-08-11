# cadybot — handoff

Context for picking this up in a fresh session. Written 2026-08-05.

Repo: <https://github.com/aarohkandy/cadybot> (public, no secrets in history —
verified). Local checkout: `~/cadybot`. Python 3.9.6, venv at `.venv`.

---

## What it is

A read-only Discord growth advisor. It logs a server to SQLite, computes a
deterministic snapshot of metrics, and an LLM turns that into stage-gated
advice. It posts in exactly one place: a private channel created by `/private`.

**Three rules everything else follows from:**

1. **The model never produces a number.** `snapshot.py` computes every metric in
   SQL. The model only interprets. This is what stops it inventing statistics.
2. **The model never grades itself.** `scorecard.py` decides whether past advice
   worked, from thresholds fixed *before* the outcome was known, and cannot
   import `advisor` or `llm`.

3. **cadybot thinks when something happens, and only then.** `agenda.py` will
   not produce a question unless a row appeared in the database, and the
   provocation's timestamp is read from that row, never from the clock —
   `next_provocation` asserts it.

Rule 2 exists because rule 1 isn't enough — a system that can't hallucinate a
statistic can still hallucinate that it's been helping. Rule 3 exists because
neither stops an agent that wakes with nothing to do and finds something anyway.

## The startup it advises

AI text-to-CAD for people who own a 3D printer but can't model what they want.
Positioning: *others make something flashy; we make something that works and
gives you what you actually asked for.*

Competitors: Zoo and AdamCAD (engineer-focused parametric CAD), Meshy/Tripo/
Sloyd (artist-focused meshes that usually need repair before printing). The gap
is functional, printable parts for hobbyists.

Where the audience is: r/3Dprinting (~3.4M, mostly solution/advice requests),
r/functionalprint, print- and model-request subs, Printables, MakerWorld.
cadybot **cannot** read those — the owner scoped it to Discord only, and reading
servers it wasn't invited to would require a self-bot, which Discord bans.

Reality check: the Discord has **1 human and 4 bots**, 0 messages in 30 days.
The product isn't finished. cadybot is no longer the bottleneck; distribution is.

## Current state

- **Running** on the owner's MacBook Air under launchd (`deploy/install.sh`),
  ~14 MB. In one server, "everything" (`1215865898523688960`).
- **Backend is local ollama** (`gemma4:e4b`) *only* because `ANTHROPIC_API_KEY`
  is empty. `CADYBOT_BACKEND=anthropic` is one line and fully wired.
- **Tests: 143 passing.** `tests/harness.py` (19 synthetic servers),
  `tests/scorer.py` (31 grading cases) and `tests/thinker.py` (93 desk cases).
  All deterministic, no model needed.
- Last commit at time of writing: `04002b2`.

### Not verified

`tests/harness.py --advice` and `tests/conversation.py` both need the model and
have **never completed a full run**. They were cut short because the owner asked
to stop using their Mac's compute. Run them after switching to the API.

## Architecture

```
config.py     env-backed settings
db.py         SQLite. Every table keyed by guild_id; no query crosses guilds.
listener.py   discord.py Client: ingest, slash commands, conversation
backfill.py   history import
snapshot.py   deterministic metrics — NO LLM EVER TOUCHES THIS FILE
stats.py      small-sample floors, pure functions
prompts.py    system prompt + per-command instructions
advisor.py    ask / brief / chat, Pydantic schemas
scorecard.py  deterministic grading of past advice — cannot import advisor/llm
ledger.py     daily close of the 8 event counts — no model, ever
agenda.py     what is worth thinking about — no model, ever
thinking.py   one pass of the desk; must never call scorecard.score
loop.py       scheduled passes: snapshot → grade → decide → narrate → deliver
llm.py        backends: ollama (local) and anthropic (Claude Opus 5)
room.py       the one private channel cadybot may write to
notify.py     the write guard — rejects every destination except that channel
playbooks/    stage-gated intervention library, loaded into the system prompt
context/      founder-maintained facts about the startup (HAS UNFILLED TODOs)
```

**Stage gates** are hard rules by member count: seed (<25), sprout (25–99),
growing (100–499), community (500+). Below 25, anything needing a crowd —
events, tournaments, XP, leaderboards, office hours — must be refused.

**Snapshot must stay roughly constant-size** regardless of server size. Lists
that grow with membership render as `{count, sample, note}`. It was 31k tokens
at 5000 members once; it's ~3.5k now, ~10-12k total prompt worst case against
`CADYBOT_OLLAMA_NUM_CTX=32768`.

## How it was built

Three coder/adversary teams (coder implements → adversarial reviewer attacks →
coder fixes), then an integration review across the combined diff. The reviews
found more than the teams did. Notable catches:

- **The live bot was broken.** Every model call runs on a worker thread via
  `asyncio.to_thread`, and SQLite refuses cross-thread connections — so `/ask`,
  `/brief` and chat were raising `ProgrammingError` in production. Fixed with
  thread-local connections.
- **`loop.py` was dead code** — never imported, so grading only ran from the CLI
  and every recommendation stayed ungraded forever.
- **The scorecard couldn't report bad news.** `not_attempted` was unreachable
  (the "did the founder act?" gate was satisfied by the founder posting
  *anything*); there was no polarity check, so `gone_quiet` rising 2→20 scored
  `worked`; a 7-member server registered success at 4→12 messages. Seven
  criticals, each fixed with a before/after probe. Those probes became
  `tests/scorer.py`.
- **A silent model downgrade** — an alternate-backend mechanism meant that on any
  week a verdict was due, the whole brief was generated by the local 4B model
  while `CADYBOT_BACKEND` said anthropic. Deleted.

## The desk (thinking without being asked)

Three tables: `ledger` (daily close of the 8 event counts), `journal` (one row
per thought, written *before* the model call so failures are charged and a crash
cannot spin), `deliveries` (what cadybot said and when — the first time anything
recorded that). Two settings keys: `agenda_installed_at`, `agenda_quiet_until`.
No `db.MIGRATIONS` entries — these are new tables, and `CREATE TABLE IF NOT
EXISTS` in `SCHEMA` is how every table here arrives.

Ceiling is `CADYBOT_THINK_CALLS_PER_DAY` (default 2) per guild per rolling day,
counted from journal rows. Each +1/day is about +$2.55/month/guild at Opus 5
pricing. `0` disables thinking entirely with no code change.

**While `CADYBOT_BACKEND=ollama` the desk thinks and journals but never speaks.**
`config.THINK_SURFACE_BACKENDS` is `("anthropic",)`. A 4B model deciding on its
own initiative to interrupt someone is the weakest link at its worst.

On the live server the honest expectation is 2-3 thoughts and 0-1 unprompted
messages *per month*. `cadybot reflect` is free and read-only — it prints what
the desk would think about right now and why it would or would not speak.

## Hard constraints

- **Discord Developer Policy §21**: message content obtained via the API must
  **never** be used to train ML models. Inference only. This is non-negotiable
  and applies to any future AI work on this data.
- **Developer Terms**: retain chat logs only as needed; delete end-user data on
  request (`python -m cadybot purge`). Never log message content.
- **Python 3.9.6** — no `match`/`case`, no `X | None` unions.
- **Privileged intents** (Server Members, Message Content) are toggles below
  10,000 reachable users; past that they need an application.
- **Administrators bypass channel privacy.** Three admin *bots* in the live
  server can read the private channel and `/remove` cannot stop them. `/who`
  says so explicitly.
- **One instance per bot token.** Two processes means two gateway sessions and
  two writers on diverging databases.

## Open decisions

**1. API key — highest leverage thing available.** Everything built is
scaffolding *around* the model. `gemma4:e4b` saying *"Events require a crowd,
which is something I cannot recommend"* is parroting the instruction, not
reasoning. Put `ANTHROPIC_API_KEY` in `.env`, set `CADYBOT_BACKEND=anthropic`,
restart. ~$2–3/month at this volume.

**2. Hosting.** Currently the owner's Mac, which sleeps — and joins/leaves during
sleep are lost permanently, being the one thing Discord won't backfill.
`deploy/linux/` has a hardened systemd unit and an idempotent `bootstrap.sh`
for a fresh Ubuntu VM. **Decision landed on a VM over Azure Container Apps**:
Container Apps is ~$15/mo vs ~$265 and scales to zero, but every project must be
a container and the build→push→deploy loop is minutes, which is friction while
experimenting. The owner wants a box to try things on.

Azure specifics: subscription is **Microsoft for Startups** with **$10k credits
expiring in ~2 years**. B-series is **not available** to it
(`NotAvailableForSubscription`) — the size list starts at $44. Chosen size:
**`D8as_v7`, 8 vCPU, 32 GB, ~$265/mo**, East US 2, Ubuntu 24.04.

A dedicated SSH key exists at `~/.ssh/cadybot_ed25519`; public half:
`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIN/pe8rLEupfLXskviGBWazm3NwlgM2Vpw4LmLTbxJm0 cadybot-vm`

**3. GPU quota — start the request early.** Founders Hub grants GPU access but
requires a separate GPU Startup classification *and* a quota increase, and
approval takes days to weeks. Credits have a clock. Never leave a GPU running:
an H100 at ~$6.98/hr is ~$5,100/month.

**Do not over-provision to "use up" credits.** An idle VM converts credits into
an idle machine, not value. GPU time is where $10k actually goes.

## Deploying to the VM (when an IP exists)

```bash
ssh -i ~/.ssh/cadybot_ed25519 azureuser@<ip>
curl -fsSL https://raw.githubusercontent.com/aarohkandy/cadybot/main/deploy/linux/bootstrap.sh | sudo bash
sudoedit /etc/cadybot/env        # DISCORD_TOKEN, ANTHROPIC_API_KEY — owner does this
sudo systemctl enable --now cadybot
journalctl -u cadybot -f
```

Then migrate the database (92 KB, holds the irreplaceable join/leave history)
with both ends stopped, and **retire the Mac agent** (`deploy/install.sh
uninstall`) so two copies aren't racing. Full steps in `deploy/linux/README.md`.

## Working agreements

- Never handle the owner's credentials. They paste tokens into `.env` themselves.
- Don't restart the live bot without saying so — a restart mid-interaction
  orphans a deferred Discord reply and surfaces as "Unknown Integration".
- The owner's Mac compute is scarce; don't run model-backed tests on it without
  asking. `ollama` holds ~3.3 GB when a model is resident.
- Adversarial review of substantial work has repeatedly paid for itself here.

## Next steps, in order

1. Add `ANTHROPIC_API_KEY`, switch backend. Biggest quality gain available.
2. Provision the VM, deploy, migrate the DB, retire the Mac agent.
3. Run `tests/harness.py --advice` and `tests/conversation.py` — still unverified.
4. Fill the TODOs in `context/product.md` (URL, what the product does well, what
   it fails at). These are the highest-leverage inputs to advice quality and only
   the owner has them.
5. Start the Azure GPU quota request if AI training is still wanted.
