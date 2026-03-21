#!/bin/bash
# run.sh - Quick start script for LLM Trading Research
# Usage:
#   ./run.sh                    # Run research loop (200 experiments)
#   ./run.sh --max 50           # Run 50 experiments
#   ./run.sh --dashboard        # Start dashboard only
#   ./run.sh --prepare          # Download/update data only
#   ./run.sh --all              # Prepare data + start dashboard + run research
#   ./run.sh --status           # Check status of running processes

set -euo pipefail
cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep '=' | xargs)
fi

# Activate venv
if [ -d .venv ]; then
    source .venv/bin/activate
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi

PYTHON="$(which python3)"

case "${1:-run}" in
    --prepare|-p)
        echo "=== Downloading and preparing data ==="
        $PYTHON prepare.py
        echo "Done. Data saved in data/processed/"
        ;;

    --dashboard|-d)
        echo "=== Starting dashboard ==="
        pkill -f "python.*dashboard.py" 2>/dev/null || true
        sleep 0.5
        $PYTHON dashboard.py &
        echo "Dashboard running at http://127.0.0.1:8888"
        ;;

    --status|-s)
        echo "=== Process Status ==="
        echo ""
        echo "Research loop:"
        ps aux | grep "agent_research" | grep -v grep || echo "  Not running"
        echo ""
        echo "Dashboard:"
        ps aux | grep "dashboard.py" | grep -v grep || echo "  Not running"
        echo ""
        echo "Results:"
        if [ -f results.tsv ]; then
            TOTAL=$(wc -l < results.tsv)
            KEPT=$(grep -c "	keep	" results.tsv 2>/dev/null || echo 0)
            echo "  Total rows: $((TOTAL - 1))"
            echo "  Kept: $KEPT"
        else
            echo "  No results yet"
        fi
        echo ""
        echo "Data:"
        if [ -d data/processed/klines ]; then
            for sym in BTCUSDT ETHUSDT SOLUSDT; do
                if [ -d "data/processed/klines/$sym" ]; then
                    tfs=$(ls "data/processed/klines/$sym/" 2>/dev/null | sed 's/.parquet//' | tr '\n' ' ')
                    echo "  $sym: $tfs"
                fi
            done
        else
            echo "  No data. Run: ./run.sh --prepare"
        fi
        echo ""
        echo "Recent log:"
        tail -5 research_loop.log 2>/dev/null || echo "  No log file"
        ;;

    --stop)
        echo "=== Stopping all processes ==="
        pkill -f "agent_research.py" 2>/dev/null && echo "Stopped research loop" || echo "Research loop not running"
        pkill -f "dashboard.py" 2>/dev/null && echo "Stopped dashboard" || echo "Dashboard not running"
        pkill -f "watchdog_loop" 2>/dev/null && echo "Stopped watchdog" || true
        ;;

    --watchdog|-w)
        echo "=== Starting watchdog (auto-restarts loop if stuck/dead) ==="
        # Start dashboard if not running
        if ! pgrep -f "dashboard.py" > /dev/null 2>&1; then
            $PYTHON dashboard.py &
            echo "Dashboard: http://127.0.0.1:8888"
        fi
        # Watchdog loop: check every 5 min, restart if dead or stuck
        WATCHDOG_INTERVAL=300  # 5 minutes
        MAX_SILENT_S=600       # 10 min without log = stuck
        echo "Watchdog: checking every ${WATCHDOG_INTERVAL}s, restart if silent > ${MAX_SILENT_S}s"
        while true; do
            if ! pgrep -f "agent_research.py" > /dev/null 2>&1; then
                echo "[$(date '+%H:%M:%S')] Loop DEAD — restarting..."
                nohup $PYTHON -u agent_research.py >> research_loop.log 2>&1 &
                echo "[$(date '+%H:%M:%S')] Restarted PID: $!"
            else
                # Check if log was updated recently
                if [ -f research_loop.log ]; then
                    LAST_MOD=$(stat -f %m research_loop.log 2>/dev/null || stat -c %Y research_loop.log 2>/dev/null)
                    NOW=$(date +%s)
                    SILENT=$((NOW - LAST_MOD))
                    if [ "$SILENT" -gt "$MAX_SILENT_S" ]; then
                        echo "[$(date '+%H:%M:%S')] Loop STUCK (no output for ${SILENT}s) — killing & restarting..."
                        pkill -9 -f "agent_research.py" 2>/dev/null
                        sleep 2
                        nohup $PYTHON -u agent_research.py >> research_loop.log 2>&1 &
                        echo "[$(date '+%H:%M:%S')] Restarted PID: $!"
                    else
                        echo "[$(date '+%H:%M:%S')] OK (last output ${SILENT}s ago)"
                    fi
                fi
            fi
            # Also keep caffeinate alive
            if ! pgrep -f "caffeinate" > /dev/null 2>&1; then
                caffeinate -d -i -s &
                echo "[$(date '+%H:%M:%S')] Restarted caffeinate"
            fi
            sleep $WATCHDOG_INTERVAL
        done
        ;;

    --all|-a)
        echo "=== Full setup: prepare + dashboard + research ==="
        # Check data
        if [ ! -d data/processed/klines ]; then
            echo "Downloading data..."
            $PYTHON prepare.py
        else
            echo "Data already exists, skipping download."
        fi
        # Dashboard
        pkill -f "python.*dashboard.py" 2>/dev/null || true
        sleep 0.5
        $PYTHON dashboard.py &
        echo "Dashboard: http://127.0.0.1:8888"
        # Research
        MAX="${2:-999999}"
        echo "Starting research loop (max=$MAX experiments)..."
        nohup $PYTHON -u agent_research.py --max "$MAX" >> research_loop.log 2>&1 &
        echo "Research loop PID: $!"
        echo "Log: tail -f research_loop.log"
        ;;

    --run|run)
        MAX="${2:-999999}"
        # Start dashboard if not running
        if ! pgrep -f "dashboard.py" > /dev/null 2>&1; then
            $PYTHON dashboard.py &
            echo "Dashboard: http://127.0.0.1:8888"
        fi
        echo "=== Starting research loop (max=$MAX experiments) ==="
        echo "Dashboard: http://127.0.0.1:8888"
        echo "Log: tail -f research_loop.log"
        nohup $PYTHON -u agent_research.py --max "$MAX" >> research_loop.log 2>&1 &
        echo "PID: $!"
        sleep 2
        tail -10 research_loop.log
        ;;

    --help|-h)
        echo "LLM Trading Research - Run Script"
        echo ""
        echo "Usage: ./run.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  (default)       Start research loop"
        echo "  --all, -a       Full setup: prepare data + dashboard + research"
        echo "  --watchdog, -w  Start with auto-restart watchdog (recommended)"
        echo "  --prepare, -p   Download/update market data"
        echo "  --dashboard, -d Start web dashboard only"
        echo "  --status, -s    Check running processes and results"
        echo "  --stop          Stop all running processes"
        echo "  --help, -h      Show this help"
        echo ""
        echo "Examples:"
        echo "  ./run.sh                # Quick start (200 experiments)"
        echo "  ./run.sh --all          # Full setup from scratch"
        echo "  ./run.sh run 500        # Run 500 experiments"
        echo "  ./run.sh --status       # Check progress"
        echo "  ./run.sh --stop         # Stop everything"
        ;;

    *)
        echo "Unknown command: $1"
        echo "Run ./run.sh --help for usage"
        exit 1
        ;;
esac
