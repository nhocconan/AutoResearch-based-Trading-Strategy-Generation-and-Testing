# Bulk Conversion Progress

- Extracted Pine files in bulk batch: `10`
- Classified bulk files: `5`
- Repo-compatible bulk Python modules backtested: `3`

## Bulk Extracted Sources

| Rank | Name | Repo TF | Pine File |
| ---: | --- | --- | --- |
| 1 | Flawless Victory Strategy - 15min BTC Machine Learning Strategy | 15m | `raw-pine/bulk/i3Uc79fF-Flawless-Victory-Strategy-15min-BTC-Machine-Learning-Strategy.pine` |
| 2 | BTC bot | 15m | `raw-pine/bulk/DPBeZbMP-BTC-bot.pine` |
| 3 |  BTC 15 min | 15m | `raw-pine/bulk/xGk5K4DE-BTC-15-min.pine` |
| 4 | Super 8 - 30M BTC | 30m | `raw-pine/bulk/zFPZofwt.pine` |
| 5 | EmperorBTC's VWAP Strategy | 5m | `raw-pine/bulk/wmJ1ovZQ-EmperorBTC-s-VWAP-Strategy.pine` |
| 6 | Automated Bitcoin (BTC) Investment Strategy from Wunderbit  | 4h | `raw-pine/bulk/0mCr8Nfv-Automated-Bitcoin-BTC-Investment-Strategy-from-Wunderbit.pine` |
| 7 | Momentum Strategy (BTC/USDT; 1h) - MACD (with source code) | 1h | `raw-pine/bulk/b7zn25L6-Momentum-Strategy-BTC-USDT-1h-MACD-with-source-code.pine` |
| 8 | Simple SuperTrend Strategy for BTCUSD 4H | 4h | `raw-pine/bulk/N0nYQBlh.pine` |
| 9 | BTC Candle Correlation Strategy  | 4h | `raw-pine/bulk/Bp053p4R-BTC-Candle-Correlation-Strategy.pine` |
| 10 | Momentum Strategy (BTC/USDT; 30m) - STOCH RSI (with source code) | 30m | `raw-pine/bulk/79Tn4cQY-Momentum-Strategy-BTC-USDT-30m-STOCH-RSI-with-source-code.pine` |

## Bulk Classifications

| Name | Classification | Reason |
| --- | --- | --- |
|  BTC 15 min | `partial` | Uses strategy.exit with tick-based profit/loss and manual intra-bar stop/target logic requiring approximation. Security function requires multi-timeframe data handling. |
| BTC bot | `unsupported` | Pine script uses security() with lookahead=true, which introduces repainting and lookahead bias. |
| Flawless Victory Strategy - 15min BTC Machine Learning Strategy | `unsupported` | depends on TradingView alert-specific behavior |
| EmperorBTC's VWAP Strategy | `unsupported` | uses higher-timeframe lookahead_on |
| Super 8 - 30M BTC | `unsupported` | depends on TradingView alert-specific behavior |

## Bulk Backtest Results

| Strategy | Symbol | TF | Return % | Sharpe | Max DD % | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| btc_15min_trend_bulk | BTCUSDT | 15m | -100.00 | -4.079 | -100.00 | 9990 |
| btc_15min_trend_bulk | ETHUSDT | 15m | -99.98 | -2.629 | -99.99 | 10320 |
| btc_investment_trend_4h_bulk | BTCUSDT | 4h | 206.61 | 0.783 | -37.03 | 106 |
| btc_investment_trend_4h_bulk | ETHUSDT | 4h | 37.70 | 0.181 | -32.29 | 116 |
| btc_macd_rsi_momentum_1h_bulk | BTCUSDT | 1h | -55.09 | -0.099 | -85.03 | 1883 |
| btc_macd_rsi_momentum_1h_bulk | ETHUSDT | 1h | -79.99 | -0.152 | -93.20 | 1912 |
