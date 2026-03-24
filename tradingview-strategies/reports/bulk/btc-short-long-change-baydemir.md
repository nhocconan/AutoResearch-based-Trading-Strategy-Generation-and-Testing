# BTC Short/Long Change Strategy [Baydemir]

- Source URL: https://www.tradingview.com/script/nafsKNWb-BTC-Short-Long-Change-Strategy-Baydemir/
- Pine file: `raw-pine/bulk/nafsKNWb-BTC-Short-Long-Change-Strategy-Baydemir.pine`
- Classification: `partial`
- Reason: Uses security() to fetch external proprietary tickers requiring custom data sourcing; logic otherwise standard
- Python file: `python-strategies/bulk/btc-short-long-change-baydemir.py`
- Timeframe: `1d`
- Import OK: `True`

## Adaptations

- Source external btcusdshorts/longs volume data
- Align multi-symbol time series data
- Implement custom WMA calculation
- Verify signal execution on bar close

## Conversion Notes

- External tickers btcusdshorts/longs substituted with volume column.
- Custom WMA implemented using numpy convolution to avoid pandas_ta.
- Signal array length matches input prices length exactly.
- Crossover logic uses shifted arrays to prevent lookahead bias.
- Module exposes name, timeframe, leverage, and generate_signals.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1d | -47.21 | -0.304 | -56.09 | 670 | 49.4 | 1.04 |
| ETHUSDT | 1d | -68.43 | -0.374 | -81.39 | 664 | 48.3 | 1.03 |
