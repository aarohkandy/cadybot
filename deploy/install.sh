#!/bin/bash
# Install cadybot as a launchd agent on this Mac.
#
#   deploy/install.sh            install and start
#   deploy/install.sh --awake    ...and stop the Mac idle-sleeping while it runs
#   deploy/install.sh restart    pick up code or .env changes
#   deploy/install.sh stop       stop until next login
#   deploy/install.sh uninstall  remove entirely
#   deploy/install.sh status     is it running?
#
# launchd starts it at login and restarts it if it dies.
#
# --awake matters more than it looks. While the Mac is asleep the bot is
# offline, and gateway events that happen meanwhile are gone for good — joins
# and leaves are the ones that cannot be backfilled from Discord afterwards.
# The cost is that the machine stops idle-sleeping, which on battery is real.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.cadybot.listener"
TARGET="gui/$(id -u)/${LABEL}"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"

case "${1:-install}" in
  install)
    [ -x "${DIR}/.venv/bin/python" ] || { echo "No venv. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
    [ -f "${DIR}/.env" ] || { echo "No .env. Copy .env.example to .env and add DISCORD_TOKEN."; exit 1; }

    mkdir -p "${DIR}/logs" "${HOME}/Library/LaunchAgents"

    AWAKE=""
    [ "${2:-}" = "--awake" ] && AWAKE="yes"
    python3 - "${DIR}" "${DIR}/deploy/${LABEL}.plist" "${PLIST}" "${AWAKE}" <<'PY'
import sys
directory, template, out, awake = sys.argv[1:5]
prefix = "        <string>/usr/bin/caffeinate</string>\n        <string>-i</string>\n" if awake else ""
body = open(template).read().replace("__CAFFEINATE__\n", prefix)
open(out, "w").write(body.replace("__CADYBOT_DIR__", directory))
PY
    [ -n "${AWAKE}" ] && echo "idle sleep disabled while cadybot runs."

    launchctl bootout "${TARGET}" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "${PLIST}"
    launchctl enable "${TARGET}"
    echo "installed and started."
    echo "logs: ${DIR}/logs/cadybot.log"
    ;;

  restart)
    launchctl kickstart -k "${TARGET}"
    echo "restarted."
    ;;

  stop)
    launchctl bootout "${TARGET}" 2>/dev/null || true
    echo "stopped. it will come back at next login unless you uninstall."
    ;;

  uninstall)
    launchctl bootout "${TARGET}" 2>/dev/null || true
    rm -f "${PLIST}"
    echo "uninstalled."
    ;;

  status)
    if launchctl print "${TARGET}" >/dev/null 2>&1; then
      launchctl print "${TARGET}" | awk '/^\tstate|^\tpid|^\tlast exit/ {print}'
    else
      echo "not installed."
    fi
    ;;

  *)
    echo "usage: $0 {install|restart|stop|uninstall|status}"; exit 1
    ;;
esac
