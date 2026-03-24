# PSAR BBPT ZLSMA BTC 1min

- Source URL: https://www.tradingview.com/script/o1YNgHYa-PSAR-BBPT-ZLSMA-BTC-1min/
- Pine file: `raw-pine/bulk/o1YNgHYa-PSAR-BBPT-ZLSMA-BTC-1min.pine`
- Classification: `partial`
- Reason: Dynamic stop-loss updates via strategy.exit and session filtering require custom Python implementation.
- Python file: `python-strategies/bulk/psar-bbpt-zlsma-btc-1min.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Implement session time masking
- Manual stop-loss/take-profit state tracking
- Replace Pine indicators with pandas_ta
- Handle partial position closing logic

## Conversion Notes

- Fixed undefined variable: renamed max_sl_pct to max_sl to match error report
- Removed session filtering logic that required unavailable timezone data
- Ensured signal array length matches input prices length exactly
- Fixed TP partial close logic to not overwrite entry/exit signals
- All variables now properly defined before use in generate_signals

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1m | -99.30 | -28.794 | -99.30 | 10017 | 28.3 | 0.45 |
| ETHUSDT | 1m | -33.10 | -8.693 | -33.16 | 722 | 35.0 | 0.45 |
