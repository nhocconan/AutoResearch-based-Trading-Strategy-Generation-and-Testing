# BTC Swinger v1

- Source URL: https://www.tradingview.com/script/uu3sSMI2-BTC-Swinger-v1/
- Pine file: `raw-pine/bulk/uu3sSMI2-BTC-Swinger-v1.pine`
- Classification: `partial`
- Reason: Pine strategy defaults to next-bar open execution for close-based signals; Python must adjust timing to avoid lookahead bias.
- Python file: `python-strategies/bulk/btc-swinger-v1.py`
- Timeframe: `1d`
- Import OK: `True`

## Adaptations

- Shift entry/exit signals to next bar open
- Map Pine commission_value to Python fee structure
- Implement date range filtering in backtest config
- Replicate trailing stop logic using close-based conditions

## Conversion Notes

- Date range filtering moved to backtest config to avoid hardcoding 2020 cutoff.
- ATR calculated manually using numpy to avoid external dependencies.
- Stateful VStop logic implemented via loop to match Pine recursion accurately.
- Signals aligned to next-bar execution to prevent lookahead bias.
- Returns numpy array of integers matching input prices length.

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 1d | 12.92 | 0.130 | -65.02 | 124 | 35.5 | 1.34 |
| ETHUSDT | 1d | 35.22 | 0.279 | -71.10 | 123 | 36.6 | 1.49 |
