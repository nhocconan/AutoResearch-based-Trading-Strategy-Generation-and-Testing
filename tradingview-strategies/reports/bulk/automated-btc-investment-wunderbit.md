# Automated Bitcoin (BTC) Investment Strategy from Wunderbit 

- Source URL: https://www.tradingview.com/script/0mCr8Nfv-Automated-Bitcoin-BTC-Investment-Strategy-from-Wunderbit/
- Pine file: `raw-pine/bulk/0mCr8Nfv-Automated-Bitcoin-BTC-Investment-Strategy-from-Wunderbit.pine`
- Classification: `partial`
- Reason: Trailing stop and limit exit logic relies on Pine's intra-bar simulation which requires approximation in Python.
- Python file: `python-strategies/bulk/automated-btc-investment-wunderbit.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Replicate custom TEMA and ATR functions
- Convert stateful trailing stop logic to pandas loop
- Approximate limit orders using bar High/Low
- Handle next-bar order execution delay

## Conversion Notes

- Converted Pine Script v5 strategy to Python with pandas/numpy only
- Implemented custom TEMA, EMA, SMA, LSMA, and ATR functions without external libraries
- Trailing stop logic converted to stateful pandas loop for proper bar-by-bar calculation
- Limit order exits (TP1/TP2) approximated using close price vs actual intrabar fills
- Entry execution delayed to next bar open to avoid lookahead bias
- Signal array returns exactly len(prices) elements (1=long, 0=flat)
- Warmup period enforced to ensure indicator stability before signals
- Only LONG trades implemented as per original Pine Script logic
- Stop loss and take profit levels calculated from entry price, not position_avg_price
- All columns used are from standard klines: open, high, low, close, volume

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | 0.00 | 0.000 | 0.00 | 0 | 0.0 | 0.00 |
| ETHUSDT | 4h | 0.00 | 0.000 | 0.00 | 0 | 0.0 | 0.00 |
