# BTC strategy

- Source URL: https://www.tradingview.com/script/SLF9v09O-BTC-strategy/
- Pine file: `raw-pine/bulk/SLF9v09O-BTC-strategy.pine`
- Classification: `partial`
- Reason: Uses strategy.exit stop loss requiring next-bar approximation; uses security() for multi-timeframe data
- Python file: `python-strategies/bulk/btc-strategy.py`
- Timeframe: `1d`
- Import OK: `True`

## Adaptations

- Approximate stop-loss execution to next-bar open/close
- Resample 4h data for security() calls
- Implement time window filtering manually

## Conversion Notes

- Fixed datetime comparison error by using pd.Timestamp instead of integers.
- Adjusted execution loop to update position for next bar, preventing same-bar fills.
- Ensured generate_signals returns a numpy array matching prices length.
- Handled NaN values in indicator calculations and time window filtering.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1d | 167.84 | 0.552 | -61.08 | 59 | 42.4 | 2.49 |
| ETHUSDT | 1d | 376.99 | 0.741 | -47.20 | 74 | 45.9 | 2.24 |
