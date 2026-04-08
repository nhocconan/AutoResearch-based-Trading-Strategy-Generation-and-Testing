# Strategy: 6h_weekly_pivot_breakout_volume_1w_trend_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.513 | -4.2% | -19.7% | 149 | FAIL |
| ETHUSDT | 0.444 | +50.0% | -10.7% | 133 | PASS |
| SOLUSDT | 0.902 | +147.3% | -23.1% | 128 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.319 | +10.8% | -10.1% | 42 | PASS |
| SOLUSDT | 0.136 | +7.5% | -9.4% | 43 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
6H Weekly Pivot Breakout with Volume Confirmation
Long when price breaks above weekly R1 with expanding volume
Short when price breaks below weekly S1 with expanding volume
Exit when price crosses back to weekly pivot point
Weekly pivot levels provide key institutional levels that work in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_volume_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly pivot levels from 1d data ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot using last 5 days (approximate week)
    high_5d = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    low_5d = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    close_5d = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).mean().values
    
    pivot = (high_5d + low_5d + close_5d) / 3
    r1 = 2 * pivot - low_5d
    s1 = 2 * pivot - high_5d
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses back below pivot
            if close[i] < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above pivot
            if close[i] > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Weekly pivot breakout with volume confirmation
            if close[i] > r1_aligned[i]:
                # Break above R1 -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < s1_aligned[i]:
                # Break below S1 -> short
                position = -1
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-07 22:40
