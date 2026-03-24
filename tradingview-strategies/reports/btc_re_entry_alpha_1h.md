# Unsupported Conversion

- Source URL: https://www.tradingview.com/script/w45uet8E/
- Pine file: `raw-pine/w45uet8E.pine`
- Compatibility: `unsupported`

## Reason

- Uses request.security(..., lookahead=barmerge.lookahead_on), which leaks higher-timeframe future information into lower-timeframe bars.

## Decision

- No Python strategy was generated because that would require a dishonest lookahead-dependent translation.
