# TradingView Conversions

These Pine v6 strategy files are independent conversions of Pine-friendly `ok` strategies from:

- `independent-verification/runs/20260321-155439/`

Files:

- `supertrend_4h_v1.pine`
- `bollinger_squeeze_breakout.pine`
- `mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1.pine`

Why these two:

- both pass the current independent verification batch
- both use only chart OHLCV, so they are reproducible in TradingView
- both avoid the older MTF/cross-symbol conversion issues that produced no-trade or low-fidelity behavior
- both are more defensible on BTC than the previous `ensemble_regime_confidence_voting_15m_v2` pick

Additional one-off conversion:

- `mtf_supertrend_macd_bbw_rsi_15m_1h_4h_v1.pine`
  - this file is a safe TradingView reinterpretation of the strategy intent
  - it uses confirmed `1h` and `4h` series only
  - it does **not** preserve the Python file's direct parquet reads or its manual `i // 16` 4h index mapping

Conversion choices:

- Orders are emitted from bar-close state transitions and rely on TradingView's next-bar fill behavior with `process_orders_on_close = false`.
- Commission is set to `0.05%` per side to approximate the verifier's `0.04% fee + 0.01% spread/slippage`.
- Entry quantity is converted from target signal fraction into asset units with `strategy.equity * abs(signal) / close`, so the script uses `default_qty_type = strategy.fixed`.
- Quantity is rounded to `syminfo.mincontract` before order placement so futures/perpetual symbols do not silently reject undersized orders.
- Both scripts hard-stop with `runtime.error(...)` if loaded on the wrong chart timeframe.

Not converted from the current top ranks:

- `adaptive_regime_trend_v7`
- `adaptive_regime_trend_v9`

Those depend on extra columns such as `funding_rate` and `taker_buy_volume`, which are not available in standard TradingView chart data.
