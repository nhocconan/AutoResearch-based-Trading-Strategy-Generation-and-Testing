# BTC outperform atrategy

- Source URL: https://www.tradingview.com/script/c8wSnkUA-BTC-outperform-atrategy/
- Pine file: `raw-pine/bulk/c8wSnkUA-BTC-outperform-atrategy.pine`
- Classification: `direct`
- Reason: Simple higher-timeframe close comparison without lookahead or complex order management.
- Python file: `python-strategies/bulk/btc-outperform-atrategy.py`
- Timeframe: `1w`
- Import OK: `True`

## Adaptations

- Implement custom 3-month close resampling logic
- Ensure higher-timeframe data aligns with completed bars
- Map TradingView symbol format to exchange API identifiers

## Conversion Notes

- Fixed pandas frequency alias '3M' to '3ME' for v2.2+ compatibility.
- Preserved module-level metadata and signal generation contract.
- Ensured numpy array return type matching input length.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1w | -31.00 | -0.190 | -71.82 | 10 | 30.0 | 1.40 |
| ETHUSDT | 1w | -39.34 | -0.221 | -71.92 | 4 | 50.0 | 2.32 |
