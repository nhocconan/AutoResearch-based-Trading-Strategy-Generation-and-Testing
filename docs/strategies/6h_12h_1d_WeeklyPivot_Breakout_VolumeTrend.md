# Strategy: 6h_12h_1d_WeeklyPivot_Breakout_VolumeTrend

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.328 | +30.6% | -4.6% | 118 | PASS |
| ETHUSDT | 0.279 | +31.0% | -8.0% | 114 | PASS |
| SOLUSDT | -0.126 | +10.9% | -15.3% | 110 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.911 | -4.6% | -4.8% | 49 | FAIL |
| ETHUSDT | 0.517 | +11.2% | -4.7% | 45 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6h_12h_1d_WeeklyPivot_Breakout_VolumeTrend
Hypothesis: 6-hour breakouts from weekly pivot levels (calculated from 1d data) with 12h trend filter and volume confirmation.
Targets 6h timeframe to reduce trade frequency (target: 15-35 trades/year) while using proven weekly pivot structure.
Only takes long when price breaks above weekly R1 with volume spike and 12h uptrend, short when breaks below weekly S1 with volume spike and 12h downtrend.
Uses weekly pivots for stronger support/resistance that works in both bull and bear markets via trend filter and volume confirmation.
"""

name = "6h_12h_1d_WeeklyPivot_Breakout_VolumeTrend"
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
    
    # Volume spike: >2.0x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's data
    # Weekly high/low/close from previous 5 trading days
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot point and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 12h EMA50
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 12h EMA50
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S1 and R1 OR closes below 12h EMA50
            if (close[i] > s1_aligned[i] and close[i] < r1_aligned[i]) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S1 and R1 OR closes above 12h EMA50
            if (close[i] > s1_aligned[i] and close[i] < r1_aligned[i]) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-12 11:42
