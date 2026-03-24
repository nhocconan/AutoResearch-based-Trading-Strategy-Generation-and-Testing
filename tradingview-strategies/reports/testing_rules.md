# TradingView Testing Rules

This is the current default test policy for all TradingView strategy backtests in `tradingview-strategies/`.

## Fair-Comparison Rules

- Initial capital: `$10,000`
- Fixed order size: `$1,000`
- Fixed target position fraction in the repo backtester: `0.1`
- Every non-zero signal is normalized to:
  - long -> `+0.1`
  - short -> `-0.1`
  - flat -> `0.0`

This rule is shared across all TradingView strategy tests so the comparisons are fair.

## Data and Cost Rules

- Price data: repo processed Binance parquet data
- Funding: keep Binance funding enabled
- Fees/slippage: use repo backtest config from `config.yaml`
- Execution model: next bar open
- Start date for scored results: `2021-01-01`
- Warmup load start: `2020-01-01`
- End date: latest locally available processed bar

## Important Interpretation Rule

- Pine sizing settings such as `strategy.cash`, `default_qty_value`, or `percent_of_equity` are documented, but they are not used in the shared comparison suite.
- If exact TradingView parity is needed for one strategy, that must be run as a separate parity-check and labeled separately.

## Canonical Source

The canonical LLM-facing source of truth is:

- `tradingview-strategies/pinescript-to-python-skill/references/testing-contract.md`

The helper that enforces the sizing in code is:

- `tradingview-strategies/tools/tv_backtest_settings.py`
