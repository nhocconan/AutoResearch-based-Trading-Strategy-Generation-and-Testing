# Mapping Checklist

Use this checklist during each conversion.

## 1. Classify Inputs

- Determine whether the Pine script is a `strategy` or `indicator`.
- Extract intended timeframe.
- Note any symbol-specific assumptions.
- Check for unsupported data sources or order-management features.

## 2. Inventory Pine Features

- `ta.*` indicators
- `request.security`
- persistent state: `var`, `varip`
- bar confirmation: `barstate.*`
- order calls: `strategy.entry`, `strategy.exit`, `strategy.close`
- plotting-only code that can be ignored

## 3. Decide Conversion Class

- `direct`: fully reproducible with repo OHLCV data
- `partial`: core signal logic reproducible, but some Pine behavior must be simplified
- `unsupported`: key dependencies missing or execution would be dishonest

## 4. Translate Carefully

- strip plotting, tables, labels, and alerts
- translate only signal logic and state needed for entries/exits
- use vectorized code unless stateful trade management requires a loop
- keep target-position outputs within realistic bounds
- remove Pine-only backtest window filters in fair-comparison mode
- normalize timestamp handling to repo datetime semantics before extracting hour/day/session fields
- do not force `unit='ms'` when converting `open_time` unless the input is actually raw integer epoch milliseconds

## 5. Validate

- run the Python strategy on the requested symbol and timeframe
- execute `generate_signals(prices)` on real repo kline data before treating the conversion as usable
- verify signal array length equals price length
- confirm no lookahead patterns were introduced
- write a report explaining deviations from Pine
- if import, signal generation, or backtest fails, repair against the exact error before marking the conversion done
- cap the repair loop at 3 attempts, then write the failure to the batch error report

## Common Pitfalls

- Pine `request.security()` aligned incorrectly to lower timeframe bars
- `strategy.exit` treated like immediate fills instead of future signal changes
- Pine `strategy.cash` or `strategy.percent_of_equity` sizing copied literally into suite backtests instead of using the shared TradingView testing contract
- unsupported intrabar assumptions hidden inside wick-based stop logic
- custom chart intervals not available in repo parquet data
- scripts whose title says BTC or ETH but whose logic is generic and timeframe-incompatible
- generated code returning pandas objects or wrong-length arrays instead of `np.ndarray`
- generated code using unavailable columns or third-party libs not present in the repo contract
- generated code comparing tz-aware repo timestamps against naive literals
- generated code assuming `open_time` is integer milliseconds and doing floor-divide on datetimes
- generated code coercing already-datetime repo timestamps with `unit='ms'`
