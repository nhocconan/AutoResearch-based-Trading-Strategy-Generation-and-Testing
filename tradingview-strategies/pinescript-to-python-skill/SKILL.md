---
name: pinescript-to-python
description: Convert TradingView Pine Script strategies into repo-compatible Python strategies for this project. Use when a task involves translating open-source Pine strategy code into Python generate_signals(prices), preserving next-bar execution semantics, classifying unsupported TradingView features, and backtesting the converted strategy on the repo's BTCUSDT or ETHUSDT parquet data.
---

# Pine Script to Python

Use this skill when converting TradingView open-source strategy code into Python strategies for this repo.

## Workflow

1. Read the Pine code and classify it before translating.
2. Only do a direct conversion when the Pine logic can be reproduced from repo data.
3. Keep all generated files, reports, and temporary artifacts inside `tradingview-strategies/`.
4. Preserve intent, not Pine syntax. Convert to a clean Python `generate_signals(prices)` implementation.
5. Backtest on repo parquet data after conversion and write a report with compatibility notes.
6. Apply the canonical TradingView testing policy in [references/testing-contract.md](references/testing-contract.md) unless the task explicitly asks for a one-off parity check.

## Compatibility Gate

Direct conversion is allowed when the script uses:

- chart OHLCV
- deterministic indicators that can be reproduced with pandas/numpy
- fixed higher-timeframe requests that map to repo timeframes
- entry and exit rules expressible as target-position signals

Mark the script `unsupported` or `partial` when it depends on:

- TradingView-only chart state such as drawings, tables, alerts, labels, or UI widgets
- unavailable data such as bid/ask, footprint, delta, CVD from tick data, broker state, or DOM/order book data
- cross-symbol logic unless the external symbol data also exists in the repo and is intentionally added
- unsupported timeframes not present in repo parquet data
- Pine execution quirks that cannot be reproduced honestly without custom simulation

If a script is unsupported, do not fake parity. Save the Pine source and write a failure report instead.

## Repo Contract

- Strategy functions must fit the repo contract in `generate_signals(prices) -> np.ndarray`.
- Use only historical information available at each bar.
- Signals represent target position fractions, not one-off orders.
- Assume the engine fills at next bar open; do not add same-bar fills in strategy logic.
- Prefer repo timeframes already available in `data/processed/klines`.
- For suite comparisons, normalize all non-zero converted signals to the fixed TradingView test size defined in `testing-contract.md`.

Read [references/repo-contract.md](references/repo-contract.md) before implementing the first conversion in a batch.

## Translation Rules

- Translate `strategy(...)` settings into notes, not runtime behavior, unless they affect signal logic.
- Do not preserve Pine position sizing literally in fair-comparison mode; use the shared TradingView test sizing contract instead.
- Map Pine series to pandas/numpy arrays and avoid per-bar Python objects unless stateful logic requires a loop.
- Prefer `pd.to_datetime(prices['open_time'], utc=True)` for repo timestamps; do not blindly pass `unit='ms'` unless the data is truly raw integer epoch milliseconds.
- If session logic needs `.date()` or `.time()`, wrap each element with `pd.Timestamp(...)` first; do not call those methods directly on `numpy.datetime64`.
- Convert `ta.*` indicators into deterministic Python helpers.
- Convert `request.security()` into explicit higher-timeframe loads and alignment only if the higher timeframe exists locally.
- Convert `strategy.entry`, `strategy.close`, and `strategy.exit` into target-position state transitions that the repo backtester can consume.
- Treat stops and take-profits as signal changes generated from historical bar data, not as broker-managed orders.
- Strip Pine backtest date-window filters in fair-comparison mode; only preserve them for explicit parity-check work.
- Keep comments short and focused on any non-obvious Pine-to-Python adaptation.
- Apply the reusable failure constraints in [references/common-failure-constraints.md](references/common-failure-constraints.md) before saving any conversion.

## Repair Loop

For reusable bulk conversion, do not stop at first-pass codegen.

1. Generate the first Python conversion.
2. Validate the module contract and importability.
3. Run a real repo-data validation pass immediately after import:
   - at minimum, load local repo klines for the intended timeframe and execute `generate_signals(prices)`
   - if the pipeline is in Stage 3, then run the full backtest too
4. If import or runtime fails, feed the exact error back into a repair pass.
5. Save the repaired output and updated notes so the next batch inherits the tighter constraints.
6. Retry at most 3 total conversion attempts per script, then persist the exact failure and move on.
7. If the Qwen batch path times out or still fails after 3 attempts, escalate that script to a Codex repair pass instead of leaving it stuck in the generic failed bucket.
8. Treat provider `429`, timeout, connection, or empty-payload failures as `retryable` infrastructure failures, not as terminal strategy-conversion errors.

When a pipeline stage is clearly failing, fix the pipeline first. Do not stop to ask the user whether obvious extractor, tracking, or runtime bugs should be repaired.

For Stage 2 conversion work, do not start browser automation. Stage 2 must consume only locally cached Pine files from Stage 1.

## Batch Use

For many conversions:

1. Crawl metadata first.
2. Filter to scripts that are BTC/ETH-relevant and repo-compatible.
3. Extract Pine code.
4. Convert one script at a time, but keep the Stage 2 worker loop running continuously until the queue is exhausted or the user explicitly stops it.
5. Save a per-script report with:
   - source URL
   - compatibility class: `direct`, `partial`, or `unsupported`
   - timeframe mapping
   - major adaptations
   - backtest results or failure reason

## Parallel Worker Discipline

When parallelizing conversion across sub-agents or workers:

1. Give each worker explicit ownership of a disjoint queue shard.
2. Require every worker to reread this skill before starting.
3. Keep all outputs inside `tradingview-strategies/` and do not touch repo data outside that folder.
4. Use the repo-configured Qwen model endpoint for classify/convert/repair steps.
5. Escalate failed or timed-out scripts to a Codex worker for targeted repair when the batch model is not getting them through.
6. Enforce the compatibility gate before any codegen.
7. Enforce the repair loop on import and backtest failures.
8. Merge shard outputs back into canonical dashboard files after each worker batch.
9. Stop retrying any single script after 3 conversion attempts; persist exact failure artifacts for later inspection.

Use [references/subagent-worker-contract.md](references/subagent-worker-contract.md) as the exact worker brief.

## Reuse Notes

- Keep a reusable Python module per Pine script under `tradingview-strategies/python-strategies/`.
- Keep one markdown report per conversion under `tradingview-strategies/reports/`.
- If a Pine script relies on `lookahead_on`, classify it as `unsupported`, add it to the lookahead blacklist, and do not rescue/adapt it into a tested strategy.
- If Pine uses stop orders, trailing stops, or `process_orders_on_close`, document the exact next-bar adaptation in the report and classify the result as `partial` unless execution parity is exact.

## References

- Repo and backtest contract: [references/repo-contract.md](references/repo-contract.md)
- Pine-to-Python mapping checklist: [references/mapping-checklist.md](references/mapping-checklist.md)
- Common failure constraints: [references/common-failure-constraints.md](references/common-failure-constraints.md)
- TradingView testing contract: [references/testing-contract.md](references/testing-contract.md)
- Current batch notes: [references/current-batch-notes.md](references/current-batch-notes.md)
- Sub-agent worker contract: [references/subagent-worker-contract.md](references/subagent-worker-contract.md)
