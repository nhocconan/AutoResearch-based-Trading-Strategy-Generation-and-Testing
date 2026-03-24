# 3kilos BTC 15m

- Source URL: https://www.tradingview.com/script/M9tw82HU-3kilos-BTC-15m/
- Pine file: `raw-pine/bulk/M9tw82HU-3kilos-BTC-15m.pine`
- Classification: `partial`
- Reason: Uses strategy.exit with limit/stop requiring intra-bar OHLC simulation to match Pine execution; stateful Supertrend logic requires manual state management.
- Python file: `python-strategies/bulk/3kilos-btc-15m.py`
- Timeframe: `15m`
- Import OK: `True`

## Adaptations

- Implement OHLC-based exit simulation
- Replicate stateful Supertrend calculation
- Handle date filtering logic
- Convert TEMA/ATR functions

## Conversion Notes

- Fixed timestamp comparison error by converting open_time to milliseconds
- Added handling for pd.Timestamp, np.datetime64, and int timestamp formats
- Preserved all strategy logic and module-level variables
- Signal array length matches prices length as required

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 15m | -33.13 | -1.124 | -37.36 | 531 | 50.3 | 0.93 |
| ETHUSDT | 15m | -28.69 | -0.899 | -33.47 | 616 | 51.0 | 0.96 |
