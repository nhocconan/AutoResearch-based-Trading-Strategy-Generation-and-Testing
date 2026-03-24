# Rsi, Ema , Ma and Bollinger Bands for 1 min Btcusdt

- Source URL: https://www.tradingview.com/script/az3Tnoj1-Rsi-Ema-Ma-and-Bollinger-Bands-for-1-min-Btcusdt/
- Pine file: `raw-pine/bulk/az3Tnoj1-Rsi-Ema-Ma-and-Bollinger-Bands-for-1-min-Btcusdt.pine`
- Classification: `partial`
- Reason: Pine strategy commands require translation to Python signal logic and position management
- Python file: `python-strategies/bulk/rsi-ema-ma-bb-1m-btcusdt.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Replace strategy.entry/close with signal generation
- Implement explicit position tracking
- Correct boolean coercion in exit conditions

## Conversion Notes

- Converted Pine strategy.entry/close to stateful position tracking.
- Implemented manual RSI with RMA smoothing to match Pine behavior.
- Shifted signals by 1 bar to prevent lookahead bias.
- Removed invalid boolean logic from Pine exit conditions (standalone indicators).
- Ensured output is numpy array matching input length.
- Added module-level name, timeframe, and leverage variables.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1m | -100.00 | -6.962 | -100.00 | 27571 | 48.5 | 0.97 |
| ETHUSDT | 1m | -100.00 | -5.644 | -100.00 | 27134 | 50.0 | 0.92 |
