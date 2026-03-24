# Pyramiding BTC 5 min no security

- Source URL: https://www.tradingview.com/script/iImkk8VO-Pyramiding-BTC-5-min-no-security/
- Pine file: `raw-pine/bulk/iImkk8VO-Pyramiding-BTC-5-min-no-security.pine`
- Classification: `partial`
- Reason: Uses strategy.exit with intrabar profit/loss logic which may be approximated in Python; pyramiding requires explicit state management.
- Python file: `python-strategies/bulk/pyramiding-btc-5m-no-security.py`
- Timeframe: `5m`
- Import OK: `True`

## Adaptations

- Manually track open trades count for pyramiding
- Convert strategy.exit to OCO orders or next-bar simulation
- Implement timestamp window filtering logic

## Conversion Notes

- Fixed timestamp comparison error by converting open_time to pandas Timestamp
- Converted start/end dates to pandas Timestamp objects for proper comparison
- Fixed profit/loss calculation to use percentage multipliers instead of tick-based
- Added max(1, ...) guards for WMA lengths to prevent division by zero
- Ensured generate_signals returns numpy array with exactly len(prices) elements
- Preserved module-level name, timeframe, and leverage variables

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 5m | -0.42 | 0.162 | -63.08 | 205 | 67.8 | 1.22 |
| ETHUSDT | 5m | 69.13 | 0.394 | -68.26 | 343 | 70.3 | 1.24 |
