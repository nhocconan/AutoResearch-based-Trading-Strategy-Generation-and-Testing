# Pyramiding BTC 5 min no security

- Source URL: https://www.tradingview.com/script/iImkk8VO-Pyramiding-BTC-5-min-no-security/
- Pine file: `raw-pine/bulk/iImkk8VO-Pyramiding-BTC-5-min-no-security.pine`
- Classification: `partial`
- Timeframe: `5m`
- Attempts used: `2`
- Result: `converted`
- Reason: Stop/target logic relies on intrabar ticks requiring next-bar approximation; pyramiding requires stateful trade count simulation.

## Adaptations

- Convert tick-based stops to price percentages
- Simulate pyramiding entry count statefully
- Remove hardcoded backtest date window
- Approximate intrabar fills to next-bar signals

## Conversion Notes

- Fixed timeout by simplifying code structure and removing verbose comments
- Converted Pine EMA, WMA, HMA3, and linear regression to pandas/numpy helpers
- Pyramiding logic simulated statefully with entry price tracking array
- Stop loss (10%) and take profit (3%) converted from ticks to price percentages
- Removed hardcoded backtest date window for fair comparison mode
- Signals represent target position (1.0 = long, 0.0 = flat) with next-bar execution
- All NaN handling added for indicator warmup periods
- Returns np.ndarray with exactly len(prices) elements as required by repo contract
