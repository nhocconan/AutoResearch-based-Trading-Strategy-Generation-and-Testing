# TradingView Testing Contract

Use this document as the canonical test policy for all backtests under `tradingview-strategies/`.

## Purpose

The TradingView suite has two distinct modes:

- fair-comparison mode: compare many converted strategies under one common sizing and repo data/cost model
- parity-check mode: reproduce one specific TradingView strategy report as closely as possible

Unless a task explicitly says otherwise, `tradingview-strategies/` uses `fair-comparison mode`.

## Fair-Comparison Mode

Apply these rules to all TradingView suite backtests, reports, dashboard detail views, and pipeline reruns.

### Account and Order Size

- Initial capital: `$10,000`
- Position sizing label: `fixed_10pct_of_10k`
- Fixed order size: `$1,000`
- Target position fraction in the repo backtester: `0.1`
- Every non-zero signal is normalized to fixed direction-only sizing:
  - long signal -> `+0.1`
  - short signal -> `-0.1`
  - flat signal -> `0.0`

This rule is intentionally shared across all converted strategies so cross-strategy comparisons are fair.

### Data Source

- Price data source: repo processed Binance parquet data already used by the main project
- Funding data source: repo processed Binance funding parquet data
- Start date for scoring period: `2021-01-01`
- Warmup load start: `2020-01-01`
- End date: latest locally available processed bar

### Cost and Execution Model

- Use repo backtest config from `config.yaml`
- Keep Binance funding enabled
- Keep repo fee/slippage config enabled
- Keep next-bar execution semantics enabled
- Do not use same-bar fills
- Do not add broker-specific intrabar fills not supported by the repo engine

### Interpretation

- TradingView `strategy(...)` sizing fields such as `default_qty_type` and `default_qty_value` are not used for fair-comparison mode.
- Pine sizing metadata may still be documented in notes, but all suite backtests must use the common fixed sizing above.
- Results must be labeled with:
  - `position_sizing`
  - `position_size_fraction`
  - `position_size_usd`
  - `initial_capital_usd`

## Parity-Check Mode

Only use this mode when the task explicitly asks to match one TradingView report as closely as possible.

In parity-check mode:

- reproduce the Pine sizing model when feasible
- match the TradingView chart symbol/exchange as closely as possible
- decide explicitly whether funding should be off or on based on the charted instrument
- label the run as `parity_check`, not as a fair-comparison suite result

Do not mix parity-check results into the main TradingView comparison dashboard unless they are clearly separated.

## Required Documentation Rule

Every TradingView suite result should make it obvious that the shared test policy was used. Reports and JSON outputs should not leave sizing assumptions implicit.

## Current Canonical Implementation

The fixed-size normalization helper lives in:

- `tradingview-strategies/tools/tv_backtest_settings.py`

Any new TradingView backtest path must use that helper before calling the repo backtester.
