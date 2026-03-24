# TradingView Strategy Conversion Summary

- Tested period: `2021-01-01` through latest locally available bars
- Symbols: `BTCUSDT`, `ETHUSDT`
- Position sizing: `fixed_10pct_of_10k` (`$1,000` fixed order size on `$10,000` initial capital)
- Funding data is included where present in repo parquet data.

## Converted Strategies

| Strategy | Compatibility | Symbol | Return % | Sharpe | Max DD % | Trades |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| tv_kinetic_kalman_breakout | direct | BTCUSDT | 6.62 | -0.582 | -8.00 | 897 |
| tv_kinetic_kalman_breakout | direct | ETHUSDT | 50.15 | 0.387 | -9.59 | 800 |
| tv_macd_trend_enhancement | partial | BTCUSDT | -3.67 | -2.665 | -5.81 | 1700 |
| tv_macd_trend_enhancement | partial | ETHUSDT | -5.43 | -2.454 | -8.50 | 1730 |
| tv_quant_trend_engine_long_only_v3 | partial | BTCUSDT | 21.58 | -0.231 | -5.16 | 93 |
| tv_quant_trend_engine_long_only_v3 | partial | ETHUSDT | 9.08 | -0.561 | -6.14 | 93 |

## Unsupported

- `btc_re_entry_alpha_1h`: Uses request.security(..., lookahead=barmerge.lookahead_on), which leaks higher-timeframe future information into lower-timeframe bars.
