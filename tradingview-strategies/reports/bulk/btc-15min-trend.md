#  BTC 15 min

- Source URL: https://www.tradingview.com/script/xGk5K4DE-BTC-15-min/
- Pine file: `raw-pine/bulk/xGk5K4DE-BTC-15-min.pine`
- Classification: `partial`
- Reason: Intrabar stop/take-profit logic using high/low crosses requires approximation; manual state tracking mixed with strategy.exit.
- Python file: `python-strategies/bulk/btc-15min-trend.py`
- Timeframe: `15m`
- Import OK: `True`

## Adaptations

- Approximate intrabar SL/TP to next-bar execution
- Resample data for security() MTF calls
- Refactor manual var state tracking to event-driven logic
- Simplify pyramiding logic for standard order management

## Conversion Notes

- Converted Pine Script v4 strategy to Python with repo-compatible interface
- MTF security() calls approximated using available 15m data only
- Intrabar SL/TP logic converted to next-bar signal execution
- Manual var state tracking refactored to event-driven loop logic
- Pyramiding logic simplified for standard signal generation
- Returns numpy array with exactly len(prices) elements
- Uses only pandas/numpy with no external dependencies
- All indicators (SMA, EMA, WMA, RSI, LinReg) implemented from scratch
- Signal values: 1=long, -1=short, 0=neutral
- No lookahead or future indexing in signal generation

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 15m | -96.53 | -3.646 | -98.11 | 8872 | 42.0 | 1.04 |
| ETHUSDT | 15m | -97.21 | -2.420 | -97.44 | 8998 | 43.6 | 1.01 |
