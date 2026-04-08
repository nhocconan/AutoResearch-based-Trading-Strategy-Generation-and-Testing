# Strategy: 6h_volume_breakout_1d_trend_enhanced

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.042 | +20.3% | -16.2% | 23 | KEEP |
| ETHUSDT | 0.012 | +15.6% | -24.4% | 14 | KEEP |
| SOLUSDT | 0.843 | +166.0% | -32.8% | 16 | KEEP |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.676 | -3.3% | -15.8% | 10 | DISCARD |
| ETHUSDT | 0.141 | +7.5% | -16.9% | 10 | KEEP |
| SOLUSDT | -0.296 | -3.0% | -22.1% | 10 | DISCARD |

## Code
```python
#!/usr/bin/env python3
"""
6h Volume Spike Breakout with 1d Trend Filter - Enhanced Version
Hypothesis: Volume-confirmed breakouts from 20-period Donchian channels capture 
institutional momentum. The 1d EMA200 filter ensures trades align with higher-timeframe 
trend, reducing whipsaws in both bull and bear markets. Volume spikes require 
>2x 20-bar average to filter weak breakouts. Added median volume filter to reduce 
false signals during low-volume periods. Target: 20-40 trades/year (~80-160 total 
over 4 years) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_breakout_1d_trend_enhanced"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Spike Detector (>2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    vol_spike = volume > (vol_ma * 2.0)
    vol_healthy = volume > vol_median  # Ensure volume is above median
    
    # 1d EMA200 Trend Filter
    df_1d = get_htf_data(prices, '1d')
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_200_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 1d EMA200
            if close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1d EMA200
            if close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high + price above 1d EMA200 + volume spike + healthy volume
            if (close[i] > high_roll[i-1] and 
                close[i] > ema_200_aligned[i] and 
                vol_spike[i] and 
                vol_healthy[i]):
                position = 1
                signals[i] = 0.25
            # Short: breakout below Donchian low + price below 1d EMA200 + volume spike + healthy volume
            elif (close[i] < low_roll[i-1] and 
                  close[i] < ema_200_aligned[i] and 
                  vol_spike[i] and 
                  vol_healthy[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-08 00:04
