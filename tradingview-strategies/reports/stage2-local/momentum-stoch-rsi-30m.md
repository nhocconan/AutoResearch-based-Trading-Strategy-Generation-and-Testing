# Momentum Strategy (BTC/USDT; 30m) - STOCH RSI (with source code)

- Source URL: https://www.tradingview.com/script/79Tn4cQY-Momentum-Strategy-BTC-USDT-30m-STOCH-RSI-with-source-code/
- Pine file: `raw-pine/bulk/79Tn4cQY-Momentum-Strategy-BTC-USDT-30m-STOCH-RSI-with-source-code.pine`
- Classification: `partial`
- Timeframe: `30m`
- Attempts used: `1`
- Result: `converted`
- Reason: Uses strategy.exit with stop/limit requiring intrabar approximation; dynamic sizing relies on trade history state.

## Adaptations

- Approximate stop/limit fills using bar high/low
- Implement stateful trade history for gamble sizing
- Normalize position sizing to repo contract
- Remove webhook JSON logic
- Enforce next-bar execution semantics

## Conversion Notes

- Converted Stochastic RSI logic using pure numpy/pandas implementations
- Approximated strategy.exit stop/limit fills using bar high/low checks
- Implemented stateful gamble sizing based on simulated trade history
- Enforced next-bar execution semantics (entries use previous bar close)
- Removed webhook JSON and TV-HUB integration code (repo-incompatible)
- Normalized position sizing to target-position fractions (-1, 0, 1)
- Bars delay logic preserved for signal-based exits
- EMA trend filter applied to both long and short entries
- Stop loss and take profit levels calculated at entry and checked each bar
- Classified as partial due to intrabar SL/TP approximation
