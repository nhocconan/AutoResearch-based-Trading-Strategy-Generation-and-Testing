# Common Failure Constraints

Apply these constraints on every Pine-to-Python conversion and every repair pass.

## Repo Contract Failures

- The module must start with `#!/usr/bin/env python3`.
- The module must define `name` as a non-empty string.
- The module must define `timeframe` as a non-empty supported timeframe string.
- `generate_signals(prices)` must exist and be callable.
- `generate_signals(prices)` must return a `numpy.ndarray` with exactly `len(prices)` elements.

## Data Discipline

- Only rely on repo kline columns: `open_time`, `open`, `high`, `low`, `close`, `volume`.
- Treat `open_time` as datetime-like repo data first; do not assume it is raw epoch milliseconds.
- Do not call `pd.to_datetime(..., unit='ms')` on repo `open_time` unless the column is actually raw integer epoch milliseconds.
- If the strategy needs session/date extraction, use `pd.Timestamp(dt)` or vectorized pandas datetime accessors before calling `.date()` or `.time()`.
- Do not read files, call APIs, or import custom helpers inside the generated strategy module.
- Do not assume TradingView broker state, DOM/order-book state, labels, tables, drawings, or alerts exist.

## Execution Semantics

- Do not model same-bar order fills.
- Convert Pine stops, targets, and trailing exits into next-bar target-position changes.
- Treat `request.security(..., lookahead=barmerge.lookahead_on)` as unsupported and blacklist it; do not convert it into a tested strategy.
- Do not fake lower-timeframe or unavailable higher-timeframe data.
- For TradingView suite comparisons, normalize all non-zero signals to the canonical fixed-size testing contract in `testing-contract.md`.
- Keep Binance funding enabled in fair-comparison mode; do not silently disable it.

## Common Runtime Bugs

- Signal length mismatch versus price length.
- Returning pandas objects when the backtester expects numpy arrays.
- Comparing tz-aware repo timestamps with tz-naive `Timestamp` literals.
- Using integer time arithmetic like `open_time // 3600000` when `open_time` is datetime-like.
- Blindly coercing repo datetime values with `unit='ms'`, which can shift or corrupt timestamps.
- Calling `.date()` or `.time()` directly on `numpy.datetime64` values.
- Rolling/apply code that breaks because the callback receives ndarray input, not Series.
- `np.where` or boolean logic on arrays with shape mismatches.
- All-NaN indicator warmups causing invalid state transitions.
- Missing guards for zero division, empty arrays, or `len(prices) == 0`.
- Using unsupported third-party libraries such as `talib`, `ta`, `scipy`, `numba`, `vectorbt`, or `pandas_ta`.

## Repair Loop

- If import fails, repair against the exact import error.
- If backtest fails, repair against the exact runtime/backtest error.
- If signal generation on real repo klines fails, repair against that exact runtime error before calling the conversion done.
- Keep strategy intent intact unless the failing behavior was dishonest to begin with.
- If Pine code contains a hardcoded backtest date window, strip that window in TradingView suite fair-comparison mode unless the user explicitly requests parity-check mode.
- Never retry the same conversion more than 3 times in one batch run. Persist the exact failure reason and any failed generated code for later inspection.
- If the batch model repeatedly times out or fails after 3 attempts, escalate the script to a Codex repair pass instead of retrying the same batch path again.
- Provider throttling, timeout, connection, or empty-payload failures are `retryable`; do not count them as terminal strategy errors.
