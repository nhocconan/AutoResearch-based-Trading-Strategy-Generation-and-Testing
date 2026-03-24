# Simple SuperTrend Strategy for BTCUSD 4H

- Source URL: https://www.tradingview.com/script/N0nYQBlh/
- Pine file: `raw-pine/bulk/N0nYQBlh.pine`
- Classification: `partial`
- Reason: Dynamic stop-loss updates and breakeven logic rely on intrabar high/low simulation; process_orders_on_close requires specific handling.
- Python file: `python-strategies/bulk/supertrend-rsi-btc-4h.py`
- Timeframe: `4h`
- Import OK: `True`

## Adaptations

- Approximate dynamic stop-loss updates to next-bar checks
- Implement process_orders_on_close entry logic
- Replicate ATR-based position sizing
- Convert SuperTrend and RSI entry conditions

## Conversion Notes

- Converted SuperTrend, RSI, and ATR calculations to pure numpy/pandas
- Entry signals on SuperTrend crossover or RSI filter with trend confirmation
- ATR-based stop loss with 1.5 multiplier approximated for next-bar execution
- Take profit and breakeven logic translated to next-bar position state changes
- Signal array length matches input prices length exactly
- No lookahead bias - entries execute on signal bar, exits checked next bar
- Only uses open/high/low/close/volume columns from klines
- Stateful position tracking replaces Pine Script strategy.position_size logic

## Backtest Results

| Symbol | Timeframe | Return % | Sharpe | Max DD % | Trades | Win Rate % | Profit Factor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTCUSDT | 4h | -99.15 | -0.390 | -99.51 | 387 | 44.7 | 1.14 |
| ETHUSDT | 4h | -99.38 | -0.108 | -99.88 | 379 | 41.7 | 1.18 |
