# Independent Verification

This folder contains a standalone audit pipeline that does not use the repo's existing `backtest.py`.

Artifacts land in `independent-verification/output/`:

- `verification_results.csv`: row-level independent results for `full`, `train`, and `test`
- `verification_results.json`: JSON export of the same data
- `strategy_summary.csv`: one summary row per strategy
- `dataset_summary.json`: data coverage and gap diagnostics
- `report.md`: human-readable audit summary
- `dashboard.html`: static dashboard
- `manifest.json`: basic run metadata
- `deep_dense_prefix.csv`: denser HTF boundary prefix-audit rows
- `execution_alignment_samples.csv`: sample proof that fills are one bar after signals
- `execution_alignment_summary.csv`: per strategy-symbol execution alignment summaries
- `cross_asset_claim_audit.csv`: audit rows for strategies claiming cross-symbol logic
- `deep_checks_summary.csv`: condensed deep-audit summary
- `deep_checks_report.md`: human-readable deep-audit report
- `deep_checks_manifest.json`: deep-audit metadata

Run it with:

```bash
./.venv/bin/python independent-verification/run_verification.py
```

Convenience wrapper:

```bash
./independent-verification/run.sh
```

What `run.sh` does:

- scans the current `strategies/*.py` set, so newly generated strategies are included automatically
- writes a timestamped run into `independent-verification/runs/YYYYMMDD-HHMMSS/`
- runs `deep_checks.py` on that same timestamped run unless skipped
- copies the newest artifacts into `independent-verification/output/`
- records the latest timestamped path in `independent-verification/latest_run.txt`

Optional flags:

```bash
./.venv/bin/python independent-verification/run_verification.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT \
  --lookahead-symbol BTCUSDT \
  --start-date 2021-01-01
```

Wrapper with custom worker count:

```bash
IV_WORKERS=6 ./independent-verification/run.sh
```

Recent-strategy remediation pass:

```bash
./.venv/bin/python verification_remediation.py --recent-limit 25
```

Scheduled integration:

- `auto_concept_research.sh` runs concept discovery first and then `verification_remediation.py` in the same scheduled job.
- If remediation finds invalid saved results, it writes `logs/restart_agent_research.flag` so `watchdog.sh` restarts the live research loop with patched logic.
- The restart is resume-safe because `agent_research.py` restores the next experiment number from persisted state plus `results.db`.

Fast mode without deep checks:

```bash
IV_SKIP_DEEP=1 ./independent-verification/run.sh
```

Audit rules implemented here:

- Load OHLCV from `data/processed/klines/{SYMBOL}/{TIMEFRAME}.parquet`
- Merge funding as an auxiliary input column when available
- Generate signals directly from each strategy's `generate_signals(prices)`
- Enforce independent execution with `signal[t] -> fill at t+1 open`
- Charge `0.05%` per side on position change
- Check signal validity and prefix stability for look-ahead
- Run denser HTF-boundary prefix checks on suspicious or high-Sharpe strategies
- Emit sample rows proving next-bar execution timing
- Flag cross-symbol claims that are not backed by observed external data reads
- Compare train/test outputs against `results.tsv`

The repo-level remediation helper adds:

- audit the most recent saved strategies from `results.db`
- purge critical failures from `results.db`, `results.tsv`, and `docs/strategies/`
- rerun stale or purged strategies with the current engine
- re-audit rerun strategies so failed fixes are removed again
- request a live autoresearch restart when invalid saved results imply the running process must reload fixed logic
