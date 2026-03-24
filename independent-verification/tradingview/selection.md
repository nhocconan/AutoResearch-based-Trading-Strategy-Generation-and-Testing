# Conversion Selection

Source run:

- `independent-verification/runs/20260321-155439/`

Selection rule:

- choose from `strategy_summary.csv`
- keep only `severity = ok`
- require `lookahead_pass = true`
- exclude `uses_fixed_date_range = true`
- exclude `uses_synthetic_open_from_close = true`
- exclude `mentions_cross_asset = true`
- exclude strategies that require non-OHLCV columns unavailable in TradingView
- prefer same-timeframe strategies over complex MTF when Pine fidelity is the priority

Chosen strategies:

1. `supertrend_4h_v1`
   - avg full sharpe: `0.295195`
   - avg full return: `78.200075%`
   - avg full max dd: `-31.343189%`
   - note: simplest verified trend-following candidate, and it stays positive on BTC in both full-period and test slices
2. `bollinger_squeeze_breakout`
   - avg full sharpe: `-1.231784`
   - avg full return: `1.949539%`
   - avg full max dd: `-8.448248%`
   - note: weak globally, but it remains one of the few Pine-friendly verified strategies that is still positive on BTC in both full-period and test slices

Not selected even though ranked higher:

- `adaptive_regime_trend_v9`
  - uses `funding_rate` and taker-volume features that standard TradingView charts do not expose
- `adaptive_regime_trend_v7`
  - same limitation as `adaptive_regime_trend_v9`
- `ensemble_regime_confidence_voting_15m_v2`
  - Pine-friendly, but independent BTC results are negative, so it is a poor TradingView candidate when the chart target is BTC
