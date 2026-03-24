# BTC Daily Strategy BF

- Source URL: https://www.tradingview.com/script/Gyy3Q3tl-BTC-Daily-Strategy-BF/
- Pine file: `raw-pine/bulk/Gyy3Q3tl-BTC-Daily-Strategy-BF.pine`
- Classification: `partial`
- Reason: Stop loss logic uses strategy.exit which requires approximation to next-bar signals in Python OHLC backtesting
- Python file: `python-strategies/bulk/btc-daily-strategy-bf.py`
- Timeframe: `1d`
- Import OK: `True`

## Adaptations

- Replace strategy.entry/exit with Python order logic
- Implement stop-loss price calculation manually
- Replicate Pine indicator calculations (EMA, RSI, Stoch)
- Apply backtest date range filtering

## Conversion Notes

- Converted Pine indicators (EMA, RSI, StochRSI, SMA) to pandas/numpy.
- Approximated strategy.exit stop-loss using iterative state tracking.
- Shifted signals by 1 bar to prevent lookahead bias on execution.
- Implemented 2017-2019 date filter using open_time column.
- Returns numpy array of positions (1, -1, 0) matching input length.
