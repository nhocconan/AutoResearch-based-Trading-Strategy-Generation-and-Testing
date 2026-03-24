# BTC Candle Correlation Strategy 

- Source URL: https://www.tradingview.com/script/Bp053p4R-BTC-Candle-Correlation-Strategy/
- Pine file: `raw-pine/bulk/Bp053p4R-BTC-Candle-Correlation-Strategy.pine`
- Classification: `unsupported`
- Timeframe: `4h`
- Attempts used: `1`
- Result: `unsupported`
- Reason: Uses multi-exchange BTC OHLC via `security()` symbols (BITFINEX/POLONIEX/BITSTAMP/COINBASE/BITMEX/KRAKEN/BINANCE); required cross-symbol/exchange data is not available in the local repo data contract, so conversion would be dishonest.

## Adaptations

- Would require adding synchronized OHLC datasets for all referenced exchanges/symbols at runtime timeframe.
- Would require deterministic cross-exchange alignment logic before signal computation.
- Pine date-window inputs can be mapped, but core signal source still depends on unavailable external feeds.
