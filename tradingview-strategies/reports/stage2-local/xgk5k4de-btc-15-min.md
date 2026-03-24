#  BTC 15 min

- Source URL: https://www.tradingview.com/script/xGk5K4DE-BTC-15-min/
- Pine file: `raw-pine/bulk/xGk5K4DE-BTC-15-min.pine`
- Classification: `unsupported`
- Timeframe: `15m`
- Attempts used: `1`
- Result: `unsupported`
- Reason: Uses security() to request lower timeframe (1m) data on 15m chart; relies on strategy.equity for sizing.

## Adaptations

- Remove lower-timeframe security request
- Replace dynamic equity sizing with fixed fraction
- Convert strategy.exit stops to signal logic
