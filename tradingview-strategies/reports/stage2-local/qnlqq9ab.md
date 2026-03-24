# ZigZag Elliott Wave Strategy (Demo)

- Source URL: https://www.tradingview.com/script/qNLQQ9aB/
- Pine file: `raw-pine/all/qNLQQ9aB.pine`
- Classification: `partial`
- Timeframe: `4h`
- Attempts used: `1`
- Result: `converted`
- Reason: Uses lag-confirmed pivots and pyramiding/partial-close behavior that must be approximated in the repo target-position signal model.

## Adaptations

- Implement ta.pivothigh/ta.pivotlow as delayed confirmation pivots without lookahead
- Approximate repeated Wave 4 strategy.entry adds as capped position scaling
- Approximate strategy.close qty_percent partial close as target-position reduction
- Map strategy.close_all exit to full flat position when wave3 target is reached

## Conversion Notes

- Converted as partial because Pine order-ticket semantics are richer than a single position-intent array.
- Wave logic follows Pine structure: Wave 2 zone entry, Wave 4 add, partial TP, full exit.
- Pivot outputs are emitted only after right-side bars confirm, matching non-lookahead behavior.
- No external imports beyond numpy/pandas and no file/API access in strategy module.
- `generate_signals` returns numeric numpy.ndarray with exact input length.

## Validation

- Module name: `ZigZag Elliott Wave Strategy (Demo)`
- Module timeframe: `4h`
- Validation symbol: `BTCUSDT`
- Bars checked: `907`
- Signal length: `907`
- Non-zero signals: `705`
- Finite ratio: `1.0`
- Signal range: `[0.0, 1.5]`
