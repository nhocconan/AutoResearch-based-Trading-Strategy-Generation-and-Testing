#!/bin/bash
# watchdog.sh - Keeps agent_research.py always running.
# Run once: nohup bash watchdog.sh > logs/watchdog.log 2>&1 &

cd "$(dirname "$0")"
export $(grep -v "^#" .env | grep "=" | xargs) 2>/dev/null
source .venv/bin/activate

INTERVAL=60   # check every 60s
LOG=logs/research.log
STUCK_AFTER=900  # restart if no output for 15 min
RESTART_FLAG=logs/restart_agent_research.flag
BOOT_TS_FILE=logs/agent_research_boot_ts
WATCH_FILES=(
    research_rules.py
    AGENT.md
    agent_research.py
    validator.py
    results_db.py
    backtest.py
    prepare.py
    mtf_data.py
    llm_client.py
    auto_concept_research.py
    auto_concept_research.sh
    auto_process_review.py
    internet_strategy_discovery.py
    STRATEGY_RULES.md
)

mkdir -p logs

get_pids() {
    pgrep -f "python3 -u agent_research.py --max 999999" || true
}

latest_watch_mtime() {
    local latest=0
    local file mtime
    for file in "${WATCH_FILES[@]}"; do
        [ -f "$file" ] || continue
        mtime=$(stat -c %Y "$file" 2>/dev/null || stat -f %m "$file" 2>/dev/null || echo 0)
        if [ "${mtime:-0}" -gt "$latest" ]; then
            latest=$mtime
        fi
    done
    echo "$latest"
}

start_agent() {
    local boot_ts
    boot_ts=$(date +%s)
    nohup env AGENT_RESEARCH_BOOT_TS="$boot_ts" python3 -u agent_research.py --max 999999 >> "$LOG" 2>&1 &
    echo "$boot_ts" > "$BOOT_TS_FILE"
    echo "[$(date '+%F %T')] Restarted, PID=$!, boot_ts=${boot_ts}"
}

restart_agent() {
    local reason="$1"
    echo "[$(date '+%F %T')] Research loop RESTART requested — ${reason}"
    pkill -f "python3 -u agent_research.py --max 999999" 2>/dev/null || true
    sleep 3
    start_agent
    rm -f "$RESTART_FLAG"
}

echo "[$(date '+%F %T')] Watchdog started (interval=${INTERVAL}s, stuck_after=${STUCK_AFTER}s)"

while true; do
    mapfile -t PIDS < <(get_pids)

    if [ "${#PIDS[@]}" -gt 1 ]; then
        echo "[$(date '+%F %T')] Duplicate research loops detected (${PIDS[*]}) — restarting cleanly..."
        restart_agent "duplicate processes"
        sleep $INTERVAL
        continue
    fi

    if [ "${#PIDS[@]}" -eq 0 ]; then
        echo "[$(date '+%F %T')] Research loop DEAD — restarting..."
        start_agent
    else
        if [ -f "$RESTART_FLAG" ]; then
            restart_agent "$(tr '\n' ' ' < "$RESTART_FLAG" | sed 's/[[:space:]]\+/ /g')"
            sleep $INTERVAL
            continue
        fi

        BOOT_TS=$(cat "$BOOT_TS_FILE" 2>/dev/null || echo 0)
        CODE_TS=$(latest_watch_mtime)
        if [ "$CODE_TS" -gt "${BOOT_TS:-0}" ]; then
            restart_agent "core research code changed on disk"
            sleep $INTERVAL
            continue
        fi

        # Check for stuck process (no log output in STUCK_AFTER seconds)
        if [ -f "$LOG" ]; then
            LAST_MOD=$(stat -c %Y "$LOG" 2>/dev/null || stat -f %m "$LOG" 2>/dev/null)
            NOW=$(date +%s)
            SILENT=$((NOW - LAST_MOD))
            if [ "$SILENT" -gt "$STUCK_AFTER" ]; then
                echo "[$(date '+%F %T')] Research loop STUCK (${SILENT}s silent) — killing & restarting..."
                pkill -9 -f "python3 -u agent_research.py --max 999999" 2>/dev/null || true
                sleep 3
                start_agent
            fi
        fi
    fi
    sleep $INTERVAL
done
