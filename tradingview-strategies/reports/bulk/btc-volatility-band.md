# BTC Volatility Band Strategy

- Source URL: https://www.tradingview.com/script/LDCIPS8i-BTC-Volatility-Band-Strategy/
- Pine file: `raw-pine/bulk/LDCIPS8i-BTC-Volatility-Band-Strategy.pine`
- Classification: `partial`
- Reason: Uses strategy.exit with stop/limit requiring intrabar fill approximation in OHLCV; stateful position tracking needed for position_avg_price.
- Python file: `python-strategies/bulk/btc-volatility-band.py`
- Timeframe: `1d`
- Import OK: `True`

## Adaptations

- Approximate intrabar stop/limit fills using bar High/Low
- Implement event-driven state tracking for position_avg_price
- Handle America/New_York timezone for date range filtering
- Convert margin settings to Python backtest capital allocation

## Conversion Notes

- Fixed datetime64 vs float comparison error by converting open_time to float timestamps
- Added dtype check for open_time column to handle datetime64[ns] or datetime64[ms]
- Used explicit timestamp constants (ms) for date range filtering
- Preserved all strategy logic including volatility bands, filters, and stateful position tracking
- Maintained signal array length matching input prices length

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1d | -5.21 | -0.277 | -36.86 | 185 | 49.7 | 1.07 |
| ETHUSDT | 1d | -29.94 | -0.431 | -51.70 | 175 | 47.4 | 0.97 |
