# Momentum Strategy (BTC/USDT; 30m) - STOCH RSI (with source code)

- Source URL: https://www.tradingview.com/script/79Tn4cQY-Momentum-Strategy-BTC-USDT-30m-STOCH-RSI-with-source-code/
- Pine file: `raw-pine/bulk/79Tn4cQY-Momentum-Strategy-BTC-USDT-30m-STOCH-RSI-with-source-code.pine`
- Classification: `partial`
- Reason: Stop-loss/take-profit uses strategy.exit implying intrabar execution in Pine but typically approximates to next-bar in Python; includes TV-specific webhook logic requiring replacement.
- Python file: `python-strategies/bulk/momentum-stoch-rsi-30m.py`
- Timeframe: `30m`
- Import OK: `True`

## Adaptations

- Replace webhook JSON generation with direct exchange API calls
- Implement persistent state management for gamble sizing logic
- Approximate intrabar SL/TP execution to next-bar open/high/low logic
- Remove TV-HUB specific configuration variables and alert messages
- Replicate strategy.position_avg_price tracking in Python state

## Conversion Notes

- Converted Stoch RSI momentum logic from Pine Script to Python
- Replaced strategy.exit intrabar SL/TP with next-bar high/low approximation
- Removed TV-HUB webhook JSON generation and alert messages
- Implemented position tracking for gamble sizing logic
- Signal array length exactly matches input prices length
- Uses only pandas/numpy with no external dependencies
- Preserved trend filter (EMA fast/slow crossover) for long/short entries
- Bars delay logic implemented for position closing
- Gamble sizing increases position after losses up to 100% limit

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 30m | -0.77 | -2.131 | -4.84 | 65 | 44.6 | 1.09 |
| ETHUSDT | 30m | -9.69 | -1.543 | -15.11 | 118 | 41.5 | 0.86 |
