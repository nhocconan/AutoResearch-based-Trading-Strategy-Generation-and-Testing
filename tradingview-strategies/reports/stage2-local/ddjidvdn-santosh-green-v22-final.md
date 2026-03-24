# santosh Green v22 - Final

- Source URL: https://www.tradingview.com/script/DdjIDVdn-santosh-Green-v22-Final/
- Pine file: `raw-pine/all/DdjIDVdn-santosh-Green-v22-Final.pine`
- Classification: `partial`
- Timeframe: `1m`
- Attempts used: `3`
- Result: `converted`
- Reason: Uses request.security multi-timeframe alignment plus strategy.exit intrabar stop/target/trailing logic and syminfo point/tick sizing that require approximation in repo execution.

## Adaptations

- Approximate request.security(..., lookahead_off) via higher-timeframe resample plus forward-fill
- Approximate intrabar strategy.exit stop/target/trailing behavior with OHLC-triggered position changes
- Approximate syminfo pointvalue/mintick sizing using inferred market tick and pointvalue=1
- Preserve NY session windows and 4:45pm close-all logic using open_time timezone conversion

## Conversion Notes

- Pine defaults were preserved, including London 02:00-03:00 NY entry window and EMA/RSI/HTF filters
- i_exit_type input exists in Pine but is not used in source execution logic; conversion mirrors effective behavior
- VWAP is session-anchored by America/New_York day for closer behavior to intraday Pine usage
- Signals are target positions in {-1,0,1} with no lookahead and same length as input bars

## Validation

- Import: `ok`
- generate_signals: `ok`
- Symbol/timeframe used: `BTCUSDT / 1m`
- Bars checked: `217441`
- Non-zero signals: `737`
- Finite ratio: `1.0`
