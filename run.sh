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

DEFAULT_PROVIDER=$(awk -F'"' '/default_provider:/ {print $2; exit}' config.yaml)
OLLAMA_URL="${OLLAMA_BASE_URL:-$(awk -F'"' '/base_url:/ && in_ollama {print $2; exit} /ollama:/ {in_ollama=1}' config.yaml)}"
OLLAMA_API_ROOT="${OLLAMA_URL%/api/chat}"
export OLLAMA_HOME="${OLLAMA_HOME:-$(pwd)/.ollama}"

ensure_local_ollama() {
    if [ "${DEFAULT_PROVIDER:-}" != "ollama" ]; then
        return 0
    fi

    case "${OLLAMA_API_ROOT:-}" in
        http://127.0.0.1:*|http://localhost:*|http://[::1]:*)
            ;;
        *)
            return 0
            ;;
    esac

    if ! command -v ollama >/dev/null 2>&1; then
        echo "ERROR: default provider is ollama but 'ollama' is not installed."
        exit 1
    fi

    if curl -fsS "${OLLAMA_API_ROOT}/api/tags" >/dev/null 2>&1; then
        return 0
    fi

    mkdir -p logs
    mkdir -p "${OLLAMA_HOME}"
    echo "=== Starting local Ollama ==="
    nohup ollama serve >> logs/ollama-server.log 2>&1 &

    for _ in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if curl -fsS "${OLLAMA_API_ROOT}/api/tags" >/dev/null 2>&1; then
            echo "Ollama is ready at ${OLLAMA_API_ROOT}"
            return 0
        fi
    done

    echo "ERROR: Ollama did not become ready at ${OLLAMA_API_ROOT}"
    exit 1
}

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
        echo "LLM provider:"
        echo "  Default: ${DEFAULT_PROVIDER:-unknown}"
        if [ "${DEFAULT_PROVIDER:-}" = "ollama" ]; then
            if command -v ollama >/dev/null 2>&1; then
                echo "  Binary: $(command -v ollama)"
            else
                echo "  Binary: not installed"
            fi
            if curl -fsS "${OLLAMA_API_ROOT}/api/tags" >/dev/null 2>&1; then
                echo "  Ollama API: up (${OLLAMA_API_ROOT})"
            else
                echo "  Ollama API: down (${OLLAMA_API_ROOT})"
            fi
        fi
        echo ""
        echo "Research loop:"
        ps aux | grep "agent_research" | grep -v grep || echo "  Not running"
        echo ""
        echo "Dashboard:"
        ps aux | grep "dashboard.py" | grep -v grep || echo "  Not running"
        echo ""
        echo "Results (SQLite):"
        if [ -f results.db ]; then
            TOTAL=$(sqlite3 results.db "SELECT COUNT(*) FROM results;" 2>/dev/null || echo 0)
            KEPT=$(sqlite3 results.db "SELECT COUNT(*) FROM results WHERE status='keep';" 2>/dev/null || echo 0)
            STRATS=$(sqlite3 results.db "SELECT COUNT(DISTINCT strategy) FROM results;" 2>/dev/null || echo 0)
            echo "  Total rows: $TOTAL | Kept: $KEPT | Strategies: $STRATS"
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
        tmux kill-session -t research 2>/dev/null && echo "Killed tmux session 'research'" || true
        pkill -f "agent_research.py" 2>/dev/null && echo "Stopped research loop" || echo "Research loop not running"
        pkill -f "dashboard.py" 2>/dev/null && echo "Stopped dashboard" || echo "Dashboard not running"
        ;;

    --watchdog|-w)
        echo "=== Starting in tmux with auto-restart ==="
        SESSION="research"
        DIR="$(pwd)"
        ACTIVATE="source $DIR/.venv/bin/activate && export \$(grep -v '^#' $DIR/.env | grep '=' | xargs)"

        # Kill existing session forcefully
        tmux kill-session -t "$SESSION" 2>/dev/null || true
        sleep 1
        # Double-check: if session persists, force kill
        tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
        sleep 0.5

        # Create tmux session — first window is research (index 0)
        tmux new-session -d -s "$SESSION" -n research -x 200 -y 50
        tmux send-keys -t "$SESSION:0" "cd $DIR && $ACTIVATE && while true; do echo \"[\$(date '+%H:%M:%S')] Starting research loop...\"; python3 -u agent_research.py --max 999999 2>&1 | tee research_loop.log; EXIT=\$?; echo \"[\$(date '+%H:%M:%S')] Research exited (\$EXIT), restarting in 5s...\"; sleep 5; done" Enter

        # Dashboard in window 1 (auto-restart on crash)
        tmux new-window -t "$SESSION:1" -n dashboard
        tmux send-keys -t "$SESSION:1" "cd $DIR && $ACTIVATE && while true; do echo \"[\$(date '+%H:%M:%S')] Starting dashboard...\"; python3 dashboard.py 2>&1 | tee dashboard.log; EXIT=\$?; echo \"[\$(date '+%H:%M:%S')] Dashboard exited (\$EXIT), restarting in 3s...\"; sleep 3; done" Enter

        # Watchdog in window 2 (monitors stuck processes)
        tmux new-window -t "$SESSION:2" -n watchdog
        tmux send-keys -t "$SESSION:2" "cd $DIR && $ACTIVATE && echo 'Watchdog started — checks every 5min for stuck processes'; while true; do sleep 300; if [ -f research_loop.log ]; then LAST_MOD=\$(stat -c %Y research_loop.log 2>/dev/null); NOW=\$(date +%s); SILENT=\$((NOW - LAST_MOD)); if [ \"\$SILENT\" -gt 600 ]; then echo \"[\$(date '+%H:%M:%S')] Research STUCK (\${SILENT}s silent) — killing...\"; pkill -9 -f 'agent_research.py' 2>/dev/null; fi; echo \"[\$(date '+%H:%M:%S')] OK (last output \${SILENT}s ago)\"; fi; done" Enter

        # Focus on research window
        tmux select-window -t "$SESSION:0"

        echo ""
        echo "tmux session '$SESSION' created with 3 windows:"
        echo "  0:research  - auto-restart research loop"
        echo "  1:dashboard - auto-restart dashboard (http://0.0.0.0:8888)"
        echo "  2:watchdog  - kills stuck processes (>10min silent)"
        echo ""
        echo "Attach:  tmux attach -t $SESSION"
        echo "Stop:    ./run.sh --stop"
        ;;

    --all|-a)
        echo "=== Full setup: prepare + dashboard + research ==="
        ensure_local_ollama
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
        ensure_local_ollama
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
