# BTC Candle Correlation Strategy 

- Source URL: https://www.tradingview.com/script/Bp053p4R-BTC-Candle-Correlation-Strategy/
- Pine file: `raw-pine/bulk/Bp053p4R-BTC-Candle-Correlation-Strategy.pine`
- Classification: `partial`
- Reason: Relies on multiple security() calls across 7 exchanges requiring multi-source data alignment.
- Python file: `python-strategies/bulk/btc-candle-correlation.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Fetch historical data for 7 BTC exchanges
- Align timestamps across multiple datasets
- Implement synthetic candle calculation from averaged OHLC
- Replicate custom squared-value indicator logic

## Conversion Notes

- Replaced 7-exchange average with single symbol OHLC due to no-external-API constraint.
- Removed hardcoded date filters for generic reusability.
- Signals shifted to next-bar to prevent lookahead bias.
- Returns numpy array matching input length.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | 224.08 | 0.657 | -62.51 | 111 | 38.7 | 1.87 |
| ETHUSDT | 4h | 53.66 | 0.320 | -77.35 | 131 | 28.2 | 1.56 |
