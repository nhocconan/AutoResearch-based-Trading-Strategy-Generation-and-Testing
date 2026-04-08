#!/bin/bash
set -euo pipefail

cd /home/trading-llm-auto-research

mkdir -p logs
LOG_FILE=auto_concept_research.log
LOCK_FILE=logs/auto_concept_research.lock
SUCCESS_STAMP=logs/auto_concept_research.last_success

exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  if ! flock -n 9; then
    printf '[%s] self-improvement cycle already running, skip\n' "$(date '+%F %T')" >> "$LOG_FILE"
    exit 0
  fi
fi

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

printf '[%s] === Self-improvement cycle START ===\n' "$(date '+%F %T')" >> "$LOG_FILE"

.venv/bin/python3 internet_strategy_discovery.py >> "$LOG_FILE" 2>&1
.venv/bin/python3 auto_process_review.py >> "$LOG_FILE" 2>&1
.venv/bin/python3 auto_concept_research.py >> "$LOG_FILE" 2>&1
if [ "${AUTO_CONCEPT_VERIFICATION_SCOPE:-recent}" = "all-filesystem" ]; then
  .venv/bin/python3 verification_remediation.py \
    --all-filesystem \
    --workers "${IV_WORKERS:-4}" \
    >> "$LOG_FILE" 2>&1
else
  .venv/bin/python3 verification_remediation.py \
    --recent-limit "${AUTO_CONCEPT_VERIFICATION_RECENT_LIMIT:-25}" \
    --workers "${IV_WORKERS:-4}" \
    >> "$LOG_FILE" 2>&1
fi

touch "$SUCCESS_STAMP"
printf '[%s] === Self-improvement cycle DONE ===\n' "$(date '+%F %T')" >> "$LOG_FILE"
