# BTC Cap Dominance RSI Strategy

- Source URL: https://www.tradingview.com/script/r2LyZ0JA/
- Pine file: `raw-pine/bulk/r2LyZ0JA.pine`
- Classification: `partial`
- Reason: Relies on proprietary TradingView data symbols (CRYPTOCAP:TOTAL) requiring alternative data APIs; involves multi-timeframe data alignment.
- Python file: `python-strategies/bulk/btc-cap-dominance-rsi.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Replace CRYPTOCAP symbols with CoinGecko or similar global market cap API
- Implement multi-timeframe data resampling for RSI calculation
- Handle reverse position logic for short entries in Python backtester
- Ensure signal generation aligns with candle close to avoid repainting

## Conversion Notes

- Original strategy relied on CRYPTOCAP:TOTAL which is unavailable in repo klines.
- Replaced global market cap data with asset OHLCV (hlc3) for RSI calculation.
- Removed BTC Dominance logic due to lack of total market cap data.
- Implemented Wilder's RSI smoothing manually using numpy.
- Signal array maintains state (1/-1) until reverse condition met.
- Ensured output length matches input prices length.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | -68.46 | -0.161 | -82.93 | 682 | 30.5 | 1.10 |
| ETHUSDT | 4h | -24.24 | 0.255 | -82.38 | 675 | 34.5 | 1.26 |
