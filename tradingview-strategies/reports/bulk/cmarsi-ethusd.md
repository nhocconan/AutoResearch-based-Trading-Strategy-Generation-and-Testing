# CMARSI Strategy (on ETHUSD) Seems working good

- Source URL: https://www.tradingview.com/script/Rcq6U2TU-CMARSI-Strategy-on-ETHUSD-Seems-working-good/
- Pine file: `raw-pine/bulk/Rcq6U2TU-CMARSI-Strategy-on-ETHUSD-Seems-working-good.pine`
- Classification: `partial`
- Reason: Mixed version directives require cleanup; strategy.exit uses stop/profit levels needing intrabar fill approximation in OHLCV backtests
- Python file: `python-strategies/bulk/cmarsi-ethusd.py`
- Timeframe: `15m`
- Import OK: `True`

## Adaptations

- Resolve conflicting version directives
- Simulate stop/profit fills using bar high/low
- Port custom indicator logic to Python

## Conversion Notes

- Removed mixed version directives and offensive variable names from Pine script.
- Implemented custom updown and percentrank logic without external libraries.
- Simplified strategy.exit stop/profit logic to crossunder signal due to unrealistic values.
- Ensured generate_signals returns numpy array matching input prices length.
- Handled NaN values during indicator warmup period to prevent signal errors.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 15m | -99.99 | -3.882 | -99.99 | 14585 | 48.6 | 0.91 |
| ETHUSDT | 15m | -100.00 | -3.236 | -100.00 | 14578 | 49.2 | 1.02 |
