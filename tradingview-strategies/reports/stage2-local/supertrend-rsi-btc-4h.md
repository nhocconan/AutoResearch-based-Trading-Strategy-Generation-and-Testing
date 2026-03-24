# Simple SuperTrend Strategy for BTCUSD 4H

- Source URL: https://www.tradingview.com/script/N0nYQBlh/
- Pine file: `raw-pine/bulk/N0nYQBlh.pine`
- Classification: `partial`
- Timeframe: `4h`
- Attempts used: `1`
- Result: `converted`
- Reason: Core indicators are reproducible. Trade management (stops, targets, breakeven) relies on Pine intrabar simulation which must be approximated to next-bar signal changes. process_orders_on_close differs from repo next-bar open fill assumption.

## Adaptations

- Convert strategy.exit stop/limit to next-bar OHLC-based signal exits
- Align execution to next-bar open fills instead of process_orders_on_close
- Normalize dynamic equity sizing to fixed repo test sizing
- Strip plotting, alerts, and date filters for fair-comparison mode
- Implement manual state tracking for breakeven and partial profit logic

## Conversion Notes

- Converted SuperTrend, RSI, and ATR indicators to pure numpy/pandas implementation
- Trade management approximated to next-bar OHLC-based exits instead of Pine intrabar simulation
- Stripped date filters, plotting, alerts for fair-comparison mode per repo contract
- Position sizing normalized to target-position signals (1, -1, 0) instead of Pine equity-based sizing
- Breakeven and partial profit logic implemented via state tracking with next-bar execution
- process_orders_on_close adapted to next-bar open fills per repo execution semantics
- Signal array length guaranteed to match input prices length
- No lookahead - all signals use only historical data available at each bar
