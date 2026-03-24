# SuperTrend AI Adaptive - Strategy [BTC]

- Source URL: https://www.tradingview.com/script/kZVrTReu-SuperTrend-AI-Adaptive-Strategy-BTC/
- Pine file: `raw-pine/bulk/kZVrTReu-SuperTrend-AI-Adaptive-Strategy-BTC.pine`
- Classification: `partial`
- Reason: Uses strategy.exit with trailing stops requiring intrabar approximation in OHLC-based Python backtesters; entries are close-confirmed.
- Python file: `python-strategies/bulk/supertrend-ai-adaptive.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Implement manual trailing stop logic using OHLC high/low
- Convert barstate.isconfirmed to close-based signal generation
- Vectorize regime detection (ADX, ATR ratio) calculations
- Adjust commission/slippage model to match Python backtester engine

## Conversion Notes

- Converted stateful SuperTrend and ADX logic to numpy loops.
- Approximated intrabar stops using bar High/Low within signal generation.
- Hardcoded default parameters from Pine Script inputs.
- Ensured signal array length matches input prices length.
- Implemented regime detection and AI scoring logic manually.
- Returns numpy int8 array for compatibility with backtest engines.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | -85.56 | -0.560 | -86.28 | 29 | 31.0 | 0.56 |
| ETHUSDT | 4h | -99.20 | -1.233 | -99.52 | 32 | 12.5 | 0.07 |
