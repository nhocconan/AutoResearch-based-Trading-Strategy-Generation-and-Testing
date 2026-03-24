# Repo Contract

This skill targets the current repo only.

## Data Access

- Klines: `data/processed/klines/{SYMBOL}/{TIMEFRAME}.parquet`
- Funding: `data/processed/funding/{SYMBOL}/funding_rate.parquet`
- Loader functions already exist in `prepare.py`

Useful repo functions:

- `prepare.load_klines(symbol, timeframe, start_date, end_date)`
- `prepare.load_funding_rate(symbol, start_date, end_date)`
- `backtest.run_backtest(signals, prices, funding_df, bt_config, leverage)`

## Supported Symbols

- `BTCUSDT`
- `ETHUSDT`
- `SOLUSDT`

The user requested testing on BTC and ETH.

## Available Timeframes

- `1m`
- `5m`
- `15m`
- `30m`
- `1h`
- `4h`
- `6h`
- `12h`
- `1d`
- `1w`

Reject or downscope scripts that require unsupported chart intervals such as `2m`, `3m`, `10m`, or custom second-based bars.

## Strategy Shape

Expected file structure is simple:

- module-level `name`
- module-level `timeframe`
- optional `leverage`
- `generate_signals(prices: pd.DataFrame) -> np.ndarray`

`prices` usually contains:

- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- auxiliary volume columns when loaded from parquet

## Execution Model

- signal at bar `t`
- fill at bar `t + 1` open
- costs are applied in the immutable repo backtester

Do not translate Pine scripts as if they fill on the same bar unless that behavior is explicitly re-expressed as a lagged signal.

## Output Discipline

For this project run, keep all new Python strategies, reports, manifests, and logs under `tradingview-strategies/`.
