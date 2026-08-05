#!/usr/bin/env bash
# Provision cadybot on a fresh Ubuntu VM. Run as root, once.
#
#   curl -fsSL https://raw.githubusercontent.com/aarohkandy/cadybot/main/deploy/linux/bootstrap.sh | sudo bash
#
# or, after cloning:  sudo deploy/linux/bootstrap.sh
#
# Afterwards, put the token in /etc/cadybot/env and start it:
#   sudoedit /etc/cadybot/env
#   systemctl enable --now cadybot
#
# Idempotent: safe to re-run to pick up a new release.

set -euo pipefail

REPO="${CADYBOT_REPO:-https://github.com/aarohkandy/cadybot.git}"
DIR=/opt/cadybot
USER=cadybot

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 1; }

echo "== packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# python3-venv is separate on Debian/Ubuntu and its absence is the single most
# common reason this script fails halfway.
apt-get install -y -qq python3 python3-venv python3-pip git ca-certificates >/dev/null

echo "== user"
id -u "$USER" >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/$USER --shell /usr/sbin/nologin "$USER"

echo "== code"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" fetch --quiet origin
  git -C "$DIR" reset --hard --quiet origin/main
else
  git clone --quiet "$REPO" "$DIR"
fi
chown -R "$USER:$USER" "$DIR"

echo "== virtualenv"
sudo -u "$USER" python3 -m venv "$DIR/.venv"
sudo -u "$USER" "$DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$USER" "$DIR/.venv/bin/pip" install --quiet -r "$DIR/requirements.txt"

echo "== config"
install -d -m 0750 -o root -g "$USER" /etc/cadybot
if [ ! -f /etc/cadybot/env ]; then
  cat > /etc/cadybot/env <<'ENV'
# cadybot configuration. Read by systemd, so: no quotes, no export, no comments
# after a value. Restart after editing: systemctl restart cadybot

DISCORD_TOKEN=

# "anthropic" is strongly preferred on a VM. A CPU-only Azure instance runs a
# 9GB local model several times slower than a laptop with Apple Silicon, and a
# GPU instance costs more per month than the API costs per year at this volume.
CADYBOT_BACKEND=anthropic
ANTHROPIC_API_KEY=
CADYBOT_MODEL=claude-opus-5

# Only used if CADYBOT_BACKEND=ollama. Point at a host that actually has a GPU.
CADYBOT_OLLAMA_HOST=http://localhost:11434
CADYBOT_OLLAMA_MODEL=gemma4:e4b
CADYBOT_OLLAMA_NUM_CTX=32768

CADYBOT_DB=/opt/cadybot/cadybot.db
ENV
  chmod 0640 /etc/cadybot/env
  chown root:"$USER" /etc/cadybot/env
  echo "   created /etc/cadybot/env — put your token in it"
else
  echo "   /etc/cadybot/env already exists, left alone"
fi

echo "== service"
install -m 0644 "$DIR/deploy/linux/cadybot.service" /etc/systemd/system/cadybot.service
systemctl daemon-reload

echo
echo "done. next:"
echo "  sudoedit /etc/cadybot/env          # DISCORD_TOKEN and ANTHROPIC_API_KEY"
echo "  systemctl enable --now cadybot"
echo "  journalctl -u cadybot -f"
