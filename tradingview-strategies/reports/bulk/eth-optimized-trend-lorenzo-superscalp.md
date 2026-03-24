# [ETH] Optimized Trend Strategy - Lorenzo SuperScalp

- Source URL: https://www.tradingview.com/script/EXfZsOdJ-ETH-Optimized-Trend-Strategy-Lorenzo-SuperScalp/
- Pine file: `raw-pine/bulk/EXfZsOdJ-ETH-Optimized-Trend-Strategy-Lorenzo-SuperScalp.pine`
- Classification: `direct`
- Reason: Standard indicators (RSI, BB, MACD) with signal-based entries/exits. No lookahead, repainting, or intrabar stop logic detected.
- Python file: `python-strategies/bulk/eth-optimized-trend-lorenzo-superscalp.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Implement stateful bar index tracking for trade cooldown
- Simulate position flip logic (close opposite before entry)
- Verify TA library calculation parity for BB and MACD
- Ensure backtester supports short selling and position netting

## Conversion Notes

- Implemented manual RSI with Wilders smoothing to match Pine ta.rsi.
- Implemented MACD and Bollinger Bands using pandas ewm and rolling.
- Added 1-bar shift in loop (decision at i-1, signal at i) to prevent lookahead.
- Stateful loop handles trade cooldown and Long/Short alternation logic.
- Returns numpy array of length len(prices) with values 1.0, -1.0, 0.0.
- Handles NaNs in early indicator calculations gracefully.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1m | -100.00 | -3.465 | -100.00 | 14825 | 60.8 | 0.88 |
| ETHUSDT | 1m | -100.00 | -3.425 | -100.00 | 14637 | 61.1 | 0.60 |
