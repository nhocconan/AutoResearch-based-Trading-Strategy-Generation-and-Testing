#!/bin/bash
# watchdog.sh - Keeps agent_research.py always running.
# Run once: nohup bash watchdog.sh > logs/watchdog.log 2>&1 &

cd "$(dirname "$0")"
export $(grep -v "^#" .env | grep "=" | xargs) 2>/dev/null
source .venv/bin/activate

INTERVAL=60   # check every 60s
LOG=logs/research.log
STUCK_AFTER=900  # restart if no output for 15 min

echo "[$(date '+%F %T')] Watchdog started (interval=${INTERVAL}s, stuck_after=${STUCK_AFTER}s)"

while true; do
    if ! pgrep -f "agent_research.py" > /dev/null 2>&1; then
        echo "[$(date '+%F %T')] Research loop DEAD — restarting..."
        nohup python3 -u agent_research.py --max 999999 >> "$LOG" 2>&1 &
        echo "[$(date '+%F %T')] Restarted, PID=$!"
    else
        # Check for stuck process (no log output in STUCK_AFTER seconds)
        if [ -f "$LOG" ]; then
            LAST_MOD=$(stat -c %Y "$LOG" 2>/dev/null || stat -f %m "$LOG" 2>/dev/null)
            NOW=$(date +%s)
            SILENT=$((NOW - LAST_MOD))
            if [ "$SILENT" -gt "$STUCK_AFTER" ]; then
                echo "[$(date '+%F %T')] Research loop STUCK (${SILENT}s silent) — killing & restarting..."
                pkill -9 -f "agent_research.py" 2>/dev/null
                sleep 3
                nohup python3 -u agent_research.py --max 999999 >> "$LOG" 2>&1 &
                echo "[$(date '+%F %T')] Restarted, PID=$!"
            fi
        fi
    fi
    sleep $INTERVAL
done
