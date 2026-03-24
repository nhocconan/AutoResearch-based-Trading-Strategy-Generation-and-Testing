# Current Batch Notes

Use these notes when extending the TradingView conversion batch in this repo.

## Converted

- `nd8EpyQ5-Kinetic-Kalman-Breakout`
  - class: `direct`
  - primary timeframe: `15m`
  - notes: clean stateful Kalman filter plus band breakouts, no unsupported data

- `2o1g6Qo5`
  - class: `partial`
  - primary timeframe: `1h`
  - notes: long-only MACD with stop, breakeven, and trailing logic adapted to next-bar signals

- `bE05CfsO-BTC-USD-Quant-Trend-Engine-Long-Only-v3-BTC-USD-4H-Timeframe`
  - class: `partial`
  - primary timeframe: `4h`
  - notes: Friday close and stop management translated into next-bar exits

## Rejected

- `w45uet8E`
  - class: `unsupported`
  - reason: `request.security(..., lookahead=barmerge.lookahead_on)` introduces future higher-timeframe values

## Repair Learnings

- Fair-comparison mode must strip Pine hardcoded date windows from generated Python, including integer-ms start/end gates.
- Repo `open_time` should be treated as datetime-like UTC data first; generated code must not blindly use `unit='ms'` or integer floor-divide time math.
- Runtime repair should target the exact failing module and then promote the fix back into the shared skill constraints before the next batch.
- Stage 1 Pine extraction should prefer HTTP-first parsing of TradingView `tree` payloads and only fall back to browser tab clicks when the page does not embed source.
- If a script page exposes `Source code` rather than plain `Code`, the extractor must treat that as the canonical code tab.
