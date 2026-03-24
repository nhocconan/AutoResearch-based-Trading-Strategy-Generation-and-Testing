# Resume After MacOS Upgrade

Saved at: 2026-03-23T01:38:01.743946+00:00

## Scope

Only resume work inside `tradingview-strategies/`.

## Saved State

- Stage 1 Pine cache: total=5403 ok=3792 pending=0 errors=1611
- Stage 2 conversion: reports=45 unsupported=6 converted_import_ok=39 convert_errors=0
- Stage 3 backtests: bulk_rows=82 bulk_files=41 manual_supported=6 manual_unsupported=1

## Resume Commands

```bash
./.venv/bin/python tradingview-strategies/tools/run_stage1_pine_cache.py --batch-size 20 --max-agents 2 --max-retries 4
./.venv/bin/python tradingview-strategies/tools/backtest_bulk_generated.py
```

## Important Files

- `tradingview-strategies/raw-pine/cache-manifest.json`
- `tradingview-strategies/results/stage-progress.json`
- `tradingview-strategies/results/stage1-runner.json`
- `tradingview-strategies/results/shutdown-state.json`
- `tradingview-strategies/results/bulk-backtests.json`
- `tradingview-strategies/results/bulk-backtest-errors.json`
- `tradingview-strategies/results/backtest_results.json`

## Shutdown Rule

Before or after any future browser-based extraction batch, do not leave `agent-browser` or `Chrome for Testing` processes running.
