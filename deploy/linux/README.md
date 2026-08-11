# Running cadybot on a Linux VM

This gets cadybot off your laptop. It fixes three things at once: your machine
stops doing the work, the bot stops going offline every time the lid closes, and
joins and leaves stop being lost — those are the one kind of event Discord will
never let you backfill.

## Sizing, honestly

The bot process is nothing. It idles at about 2 MB of RAM and writes a few rows
to SQLite when someone talks. **Where the model runs is the only decision that
matters**, and the two halves are independent.

| What | Where | Cost |
|---|---|---|
| Bot process | Smallest VM you can rent | ~$8-15/mo on Azure B1s/B1ms |
| Inference | **Claude API** | ~$2-3/mo at this volume |
| Inference | CPU-only VM | Slower than your laptop, and needs a bigger VM |
| Inference | GPU VM (Azure NC-series) | $200-400/mo |

**Do not run the model on a CPU VM.** An Azure B-series instance has no GPU and
no Apple Silicon acceleration, so a 9 GB model that answers in ~25 s on a
MacBook takes minutes there, and needs 8 GB+ of RAM to load at all — which means
paying for a B4ms to get a worse experience than the laptop you were trying to
free. A GPU instance costs more per month than the API costs per year at this
volume.

So: `CADYBOT_BACKEND=anthropic`. The local backend stays in the code because it
is genuinely useful for offline development, not because it is the cheap option.

For the VM itself, Azure is roughly twice the price of Hetzner for the same
machine. Use Azure if you have startup credits; otherwise a €4 Hetzner box does
this comfortably. The setup below is identical on both — it is just Ubuntu.

## Setup

Create an Ubuntu 22.04 or 24.04 VM. No inbound ports need opening: cadybot makes
an outbound websocket connection to Discord and nothing connects to it. Leave
SSH restricted to your own IP.

```bash
ssh azureuser@<vm-ip>
curl -fsSL https://raw.githubusercontent.com/aarohkandy/cadybot/main/deploy/linux/bootstrap.sh | sudo bash
sudoedit /etc/cadybot/env       # DISCORD_TOKEN and ANTHROPIC_API_KEY
sudo systemctl enable --now cadybot
journalctl -u cadybot -f
```

You should see `cadybot online as cadybot#5923` within a few seconds.

## Moving the existing database across

The database holds every join and leave observed so far, and that history cannot
be rebuilt. Copy it before you switch over, with the bot stopped at both ends so
nothing is mid-write:

```bash
# on the Mac
cd ~/cadybot && deploy/install.sh stop
scp cadybot.db azureuser@<vm-ip>:/tmp/cadybot.db

# on the VM
sudo systemctl stop cadybot
sudo install -o cadybot -g cadybot -m 0644 /tmp/cadybot.db /opt/cadybot/cadybot.db
sudo systemctl start cadybot
```

Then retire the Mac agent so two copies are not both ingesting the same server
and racing each other's writes:

```bash
cd ~/cadybot && deploy/install.sh uninstall
```

**Run exactly one instance per bot token.** Two processes on one token means two
gateway sessions, duplicated slash-command handling, and two writers on
databases that will silently diverge.

## Updating

```bash
sudo /opt/cadybot/deploy/linux/bootstrap.sh   # pulls main, reinstalls deps
sudo systemctl restart cadybot
```

`bootstrap.sh` is idempotent and does a hard reset to `origin/main`, so any
local edits on the VM are discarded. Edit on your machine, push, then update.

## Operating it

```bash
systemctl status cadybot
journalctl -u cadybot -f            # live
journalctl -u cadybot --since "1 hour ago"
sudo systemctl restart cadybot      # after editing /etc/cadybot/env
```

The unit restarts on crash but gives up after 5 restarts in 5 minutes. That cap
is deliberate: Discord rate-limits reconnections, and a tight crash loop is a
good way to get a token temporarily banned. If it stops, read the journal rather
than clearing the counter.

## Backups

One file, and it is the only thing here that cannot be recreated:

```bash
sudo sqlite3 /opt/cadybot/cadybot.db ".backup '/tmp/cadybot-$(date +%F).db'"
```

Use `.backup` rather than `cp` — the database runs in WAL mode, and a plain copy
of a live WAL database can land mid-transaction.


## The desk under cron

With `CADYBOT_SCHEDULER=cron` the listener starts none of the reporting loops,
including the desk. Add a fourth timer beside the nightly and weekly ones:

```
0 */6 * * *  cd /opt/cadybot && .venv/bin/python -m cadybot think --guild <id>
```

**Do not run this alongside `CADYBOT_SCHEDULER=internal`.** Two schedulers means
two passes competing for the same daily budget. They cannot duplicate a thought
— `journal`'s UNIQUE constraint sees to that — but they can spend the ceiling on
each other and leave the real provocation unaffordable.

`cadybot reflect --guild <id>` is the read-only version: it prints what the desk
would think about and whether it would speak, without a model call or a row
written. Safe to run as often as you like.
