#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter at ${PYTHON_BIN}" >&2
  exit 1
fi

TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
RUNS_DIR="${SCRIPT_DIR}/runs"
RUN_DIR="${RUNS_DIR}/${TIMESTAMP}"
LATEST_DIR="${SCRIPT_DIR}/output"

mkdir -p "${RUN_DIR}" "${LATEST_DIR}"

WORKERS="${IV_WORKERS:-4}"
SKIP_DEEP="${IV_SKIP_DEEP:-0}"

echo "Running independent verification"
echo "Root:   ${ROOT_DIR}"
echo "Run:    ${RUN_DIR}"
echo "Latest: ${LATEST_DIR}"
echo "Workers:${WORKERS}"
echo "Deep:   $([[ "${SKIP_DEEP}" == "1" ]] && echo "skip" || echo "run")"

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_verification.py" \
  --workers "${WORKERS}" \
  --output-dir "${RUN_DIR}" \
  "$@"

cat > "${SCRIPT_DIR}/latest_run.txt" <<EOF
${RUN_DIR}
EOF

if [[ "${SKIP_DEEP}" != "1" ]]; then
  echo
  echo "Running deep checks"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/deep_checks.py" --run-dir "${RUN_DIR}"
fi

cp -f "${RUN_DIR}/"*.csv "${LATEST_DIR}/"
cp -f "${RUN_DIR}/"*.json "${LATEST_DIR}/"
cp -f "${RUN_DIR}/"*.md "${LATEST_DIR}/"
cp -f "${RUN_DIR}/"*.html "${LATEST_DIR}/"

echo
echo "Done"
echo "Timestamped run: ${RUN_DIR}"
echo "Latest report:   ${LATEST_DIR}/report.md"
echo "Latest dashboard:${LATEST_DIR}/dashboard.html"
