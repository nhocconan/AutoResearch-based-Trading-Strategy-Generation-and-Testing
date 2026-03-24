# ETH Signal 15m

- Source URL: https://www.tradingview.com/script/TWCnbkiX/
- Pine file: `raw-pine/bulk/TWCnbkiX.pine`
- Classification: `partial`
- Reason: Pine strategy.exit intrabar execution logic requires approximation in Python using bar High/Low checks.
- Python file: `python-strategies/bulk/eth-signal-15m.py`
- Timeframe: `15m`
- Import OK: `True`

## Adaptations

- Implement ATR-based stop/limit using bar High/Low checks
- Replicate Supertrend and RSI calculations
- Handle date range filtering logic
- Convert equity-based position sizing

## Conversion Notes

- Fixed datetime64 vs int comparison error by converting open_time to int64 milliseconds
- Changed loop to start from index 0 to ensure all signal elements are initialized
- Preserved all strategy logic including Supertrend, RSI, and ATR-based exits
- Signal array now has exactly len(prices) elements as required

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 15m | -70.89 | -1.554 | -73.16 | 1075 | 59.5 | 0.84 |
| ETHUSDT | 15m | -76.03 | -1.156 | -81.72 | 1039 | 62.5 | 0.86 |
