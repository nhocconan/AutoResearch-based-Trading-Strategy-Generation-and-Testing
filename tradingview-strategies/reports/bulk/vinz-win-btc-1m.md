# Vinz Win BTC – STRATEGY AUTO 1m

- Source URL: https://www.tradingview.com/script/bt8HJINE/
- Pine file: `raw-pine/bulk/bt8HJINE.pine`
- Classification: `direct`
- Reason: No lookahead or security requests; uses standard OHLC logic and fixed stop/target levels.
- Python file: `python-strategies/bulk/vinz-win-btc-1m.py`
- Timeframe: `1m`
- Import OK: `True`

## Adaptations

- Configure contract specs for syminfo.pointvalue/mintick
- Implement session filtering via datetime
- Align order execution timing to next-bar open

## Conversion Notes

- Session filter implemented using UTC hour from open_time (00:00-09:00).
- SL/TP logic converted to next-bar exit signals to avoid lookahead.
- Position sizing removed; signal indicates direction only.
- Pine syminfo/mintick approximated with fixed 5.0 price buffer.
- Stateful loop used to track positions and exits accurately.
- Returns numpy array matching input prices length.
