# PSAR BBPT ZLSMA BTC 1min

- Source URL: https://www.tradingview.com/script/o1YNgHYa-PSAR-BBPT-ZLSMA-BTC-1min/
- Pine file: `raw-pine/bulk/o1YNgHYa-PSAR-BBPT-ZLSMA-BTC-1min.pine`
- Classification: `partial`
- Timeframe: `1m`
- Attempts used: `2`
- Result: `converted`
- Reason: Core indicators reproducible but strategy.exit stop/limit logic requires next-bar approximation; session logic needs UTC mapping.

## Adaptations

- Convert strategy.exit stop/limit to next-bar target-position signals
- Map TradingView session times (America/New_York) to UTC timestamps
- Replace strategy.position_avg_price with Python state tracking
- Remove plotting and UI-specific code
- Ensure generate_signals returns numpy array matching price length

## Conversion Notes

- Fixed timeout by simplifying session logic to UTC hours without complex timezone handling
- Implemented PSAR, ZLSMA, ATR, Stochastic, Bollinger Bands from scratch using numpy
- Converted strategy.exit stop/limit logic to next-bar position tracking
- Signals returned as numpy array with exactly len(prices) elements
- Session filtering approximated using UTC hour ranges for London/NY/Tokyo/Sydney
- Position state tracked in loop for SL/TP management without lookahead
- All indicators use only historical OHLCV data available at each bar
