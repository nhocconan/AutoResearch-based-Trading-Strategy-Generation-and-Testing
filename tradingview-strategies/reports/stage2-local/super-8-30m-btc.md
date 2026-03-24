# Super 8 - 30M BTC

- Source URL: https://www.tradingview.com/script/zFPZofwt/
- Pine file: `raw-pine/bulk/zFPZofwt.pine`
- Classification: `partial`
- Timeframe: `30m`
- Attempts used: `1`
- Result: `converted`
- Reason: process_orders_on_close=true conflicts with repo next-bar open fills; trailing stops and pyramiding require stateful approximation; dynamic equity sizing must be normalized.

## Adaptations

- Strip UI elements (tables, plots, alerts)
- Convert close-bar execution to next-bar open semantics
- Approximate trailing stops as next-bar signal changes
- Normalize dynamic equity sizing to fixed target positions
- Remove backtest date filters
- Manually track average price for stop/profit logic

## Conversion Notes

- Converted Pine strategy to Python with next-bar execution semantics
- Stripped UI elements (tables, plots, alerts, webhooks)
- Removed backtest date filters for fair-comparison mode
- Normalized dynamic equity sizing to fixed target positions (+1, -1, 0)
- Approximated trailing stops as next-bar signal changes using bar high/low
- Manually track average entry price for stop/profit calculations
- All indicators implemented with pandas/numpy only (no talib, ta, pandas_ta)
- Signal array length matches input prices length exactly
- Warmup period set to max indicator length to avoid NaN signals
- Classification: partial (process_orders_on_close and pyramiding approximated)
