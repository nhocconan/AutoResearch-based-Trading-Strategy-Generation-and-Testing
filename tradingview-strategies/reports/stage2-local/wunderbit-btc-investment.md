# Automated Bitcoin (BTC) Investment Strategy from Wunderbit 

- Source URL: https://www.tradingview.com/script/0mCr8Nfv-Automated-Bitcoin-BTC-Investment-Strategy-from-Wunderbit/
- Pine file: `raw-pine/bulk/0mCr8Nfv-Automated-Bitcoin-BTC-Investment-Strategy-from-Wunderbit.pine`
- Classification: `partial`
- Timeframe: `4h`
- Attempts used: `1`
- Result: `converted`
- Reason: Stateful trailing stop logic and position_avg_price dependent exits require simulation; next-bar execution approximates Pine strategy.exit limits

## Adaptations

- Convert recursive Trail1 logic to Python loop
- Simulate entry price tracking for TP/SL levels
- Map strategy.exit limits to close signals
- Remove plotting and UI inputs
- Normalize signals to target position fractions
- Strip backtest date-window filters

## Conversion Notes

- Converted TEMA, LSMA, EMA, SMA trend line calculations to pure numpy/pandas
- Implemented Wilder's ATR smoothing method matching Pine Script behavior
- Stateful trailing stop logic converted to Python loop with proper state tracking
- Entry price tracked for dynamic TP/SL level calculations
- strategy.exit limit orders converted to close signals when price reaches levels
- Removed backtest date window filters for fair-comparison mode
- Signals represent target position fractions (0 or 1 for long-only)
- Next-bar execution semantics preserved - no same-bar fills
- Plotting, alerts, and UI inputs stripped from conversion
- All indicators use only historical OHLCV data available at each bar
