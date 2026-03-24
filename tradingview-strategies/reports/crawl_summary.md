# TradingView Crawl Summary

- Listing query: `script_type=strategies&script_access=open&sort=recent_extended`
- Pages crawled: `226`
- Open-source strategy metadata rows: `5403`
- Unique strategy URLs: `5403`
- BTC/ETH candidate rows from metadata filter: `2696`
- Crawl coverage window: `2015-09-24T17:27:04+00:00` through `2026-03-21T03:32:44+00:00`

## Files

- Full manifest: `crawl/recent-open-strategies/manifest.json`
- Flattened full metadata: `crawl/recent-open-strategies/strategies.json`
- BTC/ETH candidate subset: `crawl/recent-open-strategies/btc-eth-candidates.json`
- Crawl totals: `crawl/recent-open-strategies/crawl-summary.json`

## Conversion Scope

- Pine source extracted in this batch: `4`
- Converted and backtested in this batch: `3`
- Rejected as unsupported in this batch: `1`

## Notes

- The full TradingView crawl is metadata-complete for the requested open-source strategies listing at crawl time.
- Conversions were limited to scripts whose Pine code was extracted and reviewed in this batch.
- The reusable Pine-to-Python workflow notes live under `pinescript-to-python-skill/` for extending the conversion set.
