# ETH APO Strategy [60MIN]

- Source URL: https://www.tradingview.com/script/SEZOsi3X-ETH-APO-Strategy-60MIN/
- Pine file: `raw-pine/bulk/SEZOsi3X-ETH-APO-Strategy-60MIN.pine`
- Classification: `direct`
- Reason: No lookahead security or stop/trailing logic detected
- Python file: `python-strategies/bulk/eth-apo-strategy-60min.py`
- Timeframe: `1h`
- Import OK: `True`

## Adaptations

- Implement APO and RSI calculations
- Manage position state manually
- Align order execution timing

## Conversion Notes

- Ensured signal array length matches input prices length.
- Added module-level name, timeframe, leverage variables.
- Implemented generate_signals returning numpy array.
- Used only close column from prices DataFrame.
- Filled NaNs in indicators to prevent logic errors.
- Shifted signals by one bar to align with next-bar execution.
- Removed all lookahead bias and external dependencies.
- Implemented APO, RSI, and SMA calculations manually.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1h | -56.82 | -0.707 | -65.91 | 752 | 53.7 | 0.94 |
| ETHUSDT | 1h | 255.21 | 0.681 | -47.55 | 314 | 69.7 | 1.33 |
