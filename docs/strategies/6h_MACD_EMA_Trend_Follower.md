# Strategy: 6h_MACD_EMA_Trend_Follower

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.160 | +16.6% | -10.2% | 107 | FAIL |
| ETHUSDT | 0.061 | +23.0% | -11.0% | 94 | PASS |
| SOLUSDT | 0.637 | +66.0% | -11.8% | 113 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.216 | +8.2% | -6.6% | 28 | PASS |
| SOLUSDT | -0.415 | +2.8% | -5.2% | 25 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_MACD_EMA_Trend_Follower
# Hypothesis: Follow medium-term trend using 6h MACD histogram and EMA200 filter.
# Long when MACD histogram turns positive AND price above EMA200 (6h).
# Short when MACD histogram turns negative AND price below EMA200 (6h).
# Uses weekly trend filter to avoid counter-trend trades in strong trends.
# Designed to capture trends while minimizing whipsaws in both bull and bear markets.

name = "6h_MACD_EMA_Trend_Follower"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 6h data for MACD and EMA200
    df_6h = get_htf_data(prices, '6h')
    
    # MACD (12,26,9) on 6h close
    close_6h = df_6h['close'].values
    ema12 = pd.Series(close_6h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_6h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # EMA200 on 6h close
    ema200_6h = pd.Series(close_6h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly trend filter: price above/below weekly EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Align 6h indicators to lower timeframe (15m equivalent for 6h)
    macd_hist_aligned = align_htf_to_ltf(prices, df_6h, macd_hist)
    ema200_6h_aligned = align_htf_to_ltf(prices, df_6h, ema200_6h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(ema200_6h_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # MACD histogram crossover signals
        macd_hist_prev = macd_hist_aligned[i-1] if i > 0 else 0
        macd_hist_curr = macd_hist_aligned[i]
        
        # Bullish crossover: MACD hist crosses above zero
        bullish_cross = macd_hist_prev <= 0 and macd_hist_curr > 0
        # Bearish crossover: MACD hist crosses below zero
        bearish_cross = macd_hist_prev >= 0 and macd_hist_curr < 0
        
        # Price relative to EMA200
        price_above_ema200 = close[i] > ema200_6h_aligned[i]
        price_below_ema200 = close[i] < ema200_6h_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]

        if position == 0:
            # LONG: MACD bullish crossover + price above EMA200 + weekly uptrend
            if bullish_cross and price_above_ema200 and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: MACD bearish crossover + price below EMA200 + weekly downtrend
            elif bearish_cross and price_below_ema200 and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: MACD bearish crossover OR price below EMA200 OR weekly downtrend
            if bearish_cross or not price_above_ema200 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: MACD bullish crossover OR price above EMA200 OR weekly uptrend
            if bullish_cross or not price_below_ema200 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
```

## Last Updated
2026-05-13 02:16
