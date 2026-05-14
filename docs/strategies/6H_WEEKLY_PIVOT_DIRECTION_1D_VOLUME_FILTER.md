# Strategy: 6H_WEEKLY_PIVOT_DIRECTION_1D_VOLUME_FILTER

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.512 | +41.5% | -6.7% | 169 | PASS |
| ETHUSDT | 0.281 | +33.7% | -10.8% | 139 | PASS |
| SOLUSDT | 1.221 | +167.2% | -13.9% | 116 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.905 | -5.5% | -8.2% | 49 | FAIL |
| ETHUSDT | 0.307 | +9.0% | -6.3% | 39 | PASS |
| SOLUSDT | -1.149 | -5.4% | -13.4% | 32 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6H_WEEKLY_PIVOT_DIRECTION_1D_VOLUME_FILTER
Hypothesis: Use weekly pivot points to establish long-term bias, then trade breakouts from daily support/resistance levels in the direction of that bias, with volume confirmation. Weekly pivot bias filters out counter-trend trades, improving performance in both bull and bear markets by avoiding false breakouts. Target: 15-35 trades/year.
"""
name = "6H_WEEKLY_PIVOT_DIRECTION_1D_VOLUME_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points from previous week
    weekly_pivot = np.full(len(close_1w), np.nan)
    weekly_r1 = np.full(len(close_1w), np.nan)
    weekly_s1 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        weekly_pivot[i] = (ph + pl + pc) / 3.0
        weekly_r1[i] = 2 * weekly_pivot[i] - pl
        weekly_s1[i] = 2 * weekly_pivot[i] - ph
    
    # 1d data for daily support/resistance and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points from previous day
    daily_pivot = np.full(len(close_1d), np.nan)
    daily_r1 = np.full(len(close_1d), np.nan)
    daily_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        daily_pivot[i] = (ph + pl + pc) / 3.0
        daily_r1[i] = 2 * daily_pivot[i] - pl
        daily_s1[i] = 2 * daily_pivot[i] - ph
    
    # EMA50 for 1d trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current 6h volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    # Align all higher timeframe data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(daily_pivot_aligned[i]) or 
            np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Weekly bias: above weekly pivot = bullish bias, below = bearish bias
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # LONG: Break above daily R1 with volume spike and weekly bullish bias
            if (high[i] > daily_r1_aligned[i] and 
                volume_spike[i] and 
                weekly_bullish):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below daily S1 with volume spike and weekly bearish bias
            elif (low[i] < daily_s1_aligned[i] and 
                  volume_spike[i] and 
                  weekly_bearish):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below daily pivot or weekly bias turns bearish
            if (close[i] < daily_pivot_aligned[i] or 
                not weekly_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above daily pivot or weekly bias turns bullish
            if (close[i] > daily_pivot_aligned[i] or 
                not weekly_bearish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 10:19
