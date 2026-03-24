# Ultimate T3 Fibonacci for BTC Scalping. Look at backtest report!

- Source URL: https://www.tradingview.com/script/AcsNIzBt-Ultimate-T3-Fibonacci-for-BTC-Scalping-Look-at-backtest-report/
- Pine file: `raw-pine/bulk/AcsNIzBt-Ultimate-T3-Fibonacci-for-BTC-Scalping-Look-at-backtest-report.pine`
- Classification: `partial`
- Reason: Uses strategy.exit with limit/stop requiring intrabar fill approximation in Python; custom T3 function needs manual implementation
- Python file: `python-strategies/bulk/ultimate-t3-fibonacci-btc-scalping.py`
- Timeframe: `30m`
- Import OK: `True`

## Adaptations

- Approximate strategy.exit limit/stop using bar High/Low
- Implement custom T3 indicator calculation
- Track position average price manually for dynamic TP/SL

## Conversion Notes

- Implemented custom T3 indicator using numpy loops for EMA cascades.
- Approximated strategy.exit TP/SL using bar High/Low checks within stateful loop.
- Signals returned as numpy array matching input prices length.
- Module-level name, timeframe, and leverage variables defined.
- Date range filtering omitted for general reusability.
