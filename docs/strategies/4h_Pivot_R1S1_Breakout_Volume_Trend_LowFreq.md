# Strategy: 4h_Pivot_R1S1_Breakout_Volume_Trend_LowFreq

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.070 | +23.3% | -8.6% | 266 | PASS |
| ETHUSDT | 0.185 | +28.6% | -8.4% | 249 | PASS |
| SOLUSDT | 0.625 | +71.6% | -17.5% | 210 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.099 | -2.1% | -4.2% | 97 | FAIL |
| ETHUSDT | 0.720 | +15.6% | -9.9% | 93 | PASS |
| SOLUSDT | 0.672 | +14.8% | -8.2% | 75 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume_Trend_LowFreq
Hypothesis: 4-hour breakouts above R1 or below S1 of daily Camarilla pivots with volume confirmation and 1-day EMA trend filter.
Uses stricter volume threshold and longer EMA period to reduce trades and avoid overtrading.
Target: 15-30 trades/year on 4h timeframe with disciplined entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R1, S1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + range_1d * 1.1 / 12
    s1_1d = pivot_1d - range_1d * 1.1 / 12
    
    # Align 1-day levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1-day EMA50 trend filter (longer period for stronger trend)
    ema50_1d = np.full(len(close_1d), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[0:51])
        else:
            ema50_1d[i] = close_1d[i] * k + ema50_1d[i-1] * (1 - k)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average (stricter threshold)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and 1-day uptrend
            if (close[i] > r1_1d_aligned[i] and vol_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and 1-day downtrend
            elif (close[i] < s1_1d_aligned[i] and vol_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below pivot or 1-day trend turns down
            if (close[i] < pivot_1d_aligned[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above pivot or 1-day trend turns up
            if (close[i] > pivot_1d_aligned[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume_Trend_LowFreq"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-18 08:47
