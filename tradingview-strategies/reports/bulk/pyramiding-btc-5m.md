# Pyramiding BTC 5 min

- Source URL: https://www.tradingview.com/script/9eVcc7MB-Pyramiding-BTC-5-min/
- Pine file: `raw-pine/bulk/9eVcc7MB-Pyramiding-BTC-5-min.pine`
- Classification: `partial`
- Reason: Stop-loss/take-profit uses tick-based strategy.exit requiring intrabar approximation; pyramiding relies on strategy.opentrades state; security function needs MTF data alignment.
- Python file: `python-strategies/bulk/pyramiding-btc-5m.py`
- Timeframe: `5m`
- Import OK: `True`

## Adaptations

- Replace security with resampled data
- Approximate tick stops to bar OHLC
- Implement pyramiding state machine
- Convert tick values to price levels

## Conversion Notes

- Fixed datetime vs int comparison error by using pd.Timestamp for START_TIMESTAMP.
- Added robust timezone handling for open_time column to prevent dtype mismatches.
- Ensured hma sqrt length is explicitly cast to int.
- Preserved strategy logic, signal return format, and module-level variables.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 5m | -13.94 | 0.105 | -67.34 | 213 | 67.1 | 1.18 |
| ETHUSDT | 5m | 26.21 | 0.303 | -71.08 | 351 | 69.5 | 1.22 |
