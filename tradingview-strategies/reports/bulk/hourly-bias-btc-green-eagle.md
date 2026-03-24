# Hourly Bias on BTC in Bullish USA Session “Green Eagle”

- Source URL: https://www.tradingview.com/script/97ou6vBe-Hourly-Bias-on-BTC-in-Bullish-USA-Session-Green-Eagle/
- Pine file: `raw-pine/bulk/97ou6vBe-Hourly-Bias-on-BTC-in-Bullish-USA-Session-Green-Eagle.pine`
- Classification: `direct`
- Reason: Simple time-based entry/exit with ATR filter, no lookahead or intrabar stop logic.
- Python file: `python-strategies/bulk/hourly-bias-btc-green-eagle.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Map Pine dayofweek enum to Python weekday
- Align exchange timezone for hour checks
- Configure commission/slippage in backtester

## Conversion Notes

- Fixed AttributeError by using .dt accessor on pandas datetime Series for hour/weekday extraction.
- Preserved signal array length contract (len(prices)).
- Maintained UTC timezone alignment for hour checks.
- Kept strategy logic unchanged regarding ATR calculation method (SMA) to minimize deviations.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | -10.22 | -0.390 | -29.82 | 259 | 46.7 | 1.03 |
| ETHUSDT | 1h | 24.64 | 0.060 | -29.84 | 250 | 54.8 | 1.25 |
