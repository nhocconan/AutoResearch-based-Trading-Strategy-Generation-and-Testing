# BTC Intraday Advanced Spot PRO V6

- Source URL: https://www.tradingview.com/script/k0yPTqHH-BTC-Intraday-Advanced-Spot-PRO-V6/
- Pine file: `raw-pine/bulk/k0yPTqHH-BTC-Intraday-Advanced-Spot-PRO-V6.pine`
- Classification: `partial`
- Reason: Uses calc_on_every_tick=true requiring intrabar approximation; strategy.exit stop/limit orders need OHLCV simulation.
- Python file: `python-strategies/bulk/btc-intraday-advanced-spot-pro-v6.py`
- Timeframe: `5m`
- Import OK: `True`

## Adaptations

- Replace calc_on_every_tick with bar close or high/low checks
- Simulate strategy.exit SL/TP using bar High/Low ranges
- Adjust Break Even logic to trigger on High/Low breach instead of close
- Remove plotting and overlay specific code

## Conversion Notes

- Converted Pine Script v6 to Python with pandas/numpy only.
- Replaced calc_on_every_tick with bar close entry and High/Low exit simulation.
- Simplified partial TP1 exit to Break Even activation logic for signal compatibility.
- Signals returned as numpy array (1, -1, 0) matching input length.
- No lookahead: exits evaluated on current bar OHLC, entries on close.
- Module exposes name, timeframe, leverage, and generate_signals.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 5m | -97.13 | -2.516 | -97.27 | 6335 | 31.2 | 0.95 |
| ETHUSDT | 5m | -96.97 | -2.264 | -97.97 | 7440 | 32.7 | 0.99 |
