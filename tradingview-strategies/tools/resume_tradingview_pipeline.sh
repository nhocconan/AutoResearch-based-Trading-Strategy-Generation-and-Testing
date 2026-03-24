#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
./.venv/bin/python tradingview-strategies/tools/run_stage1_pine_cache.py --batch-size 20 --max-agents 2 --max-retries 4
