# Strategy: 6h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.199 | +13.4% | -15.5% | 195 | FAIL |
| ETHUSDT | 0.156 | +27.2% | -13.9% | 174 | PASS |
| SOLUSDT | 0.332 | +42.5% | -13.6% | 133 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.538 | +26.5% | -5.8% | 57 | PASS |
| SOLUSDT | -0.066 | +4.8% | -12.9% | 50 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1
Concept: Conservative pivot breakout with volume confirmation and reduced frequency to combat overtrading.
- Long: Close > R1 AND volume > 2.0x 50-period average (stricter volume filter)
- Short: Close < S1 AND volume > 2.0x 50-period average
- Exit: Price returns to pivot point (PP)
- Uses 50-period volume average for stability and fewer signals
- Position sizing: 0.25 (conservative to limit drawdown)
- Target: <100 total trades over 4 years to minimize fee drag
- Works in bull/bear: Pivots adapt to market structure, high volume threshold filters false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R1S1_Breakout_Volume_Conservative_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === 1d: Calculate Pivot Points (Standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*PP - L
    r1_1d = 2 * pp_1d - low_1d
    # S1 = 2*PP - H
    s1_1d = 2 * pp_1d - high_1d
    
    # Align pivot levels to 6h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 6h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 50-period average for stability and fewer signals
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Get values
        pp_val = pp_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0x 50-period average (much stricter)
        vol_condition = current_volume > 2.0 * current_vol_ma
        
        if position == 0:
            # Long: close above R1 with high volume confirmation
            if current_close > r1_val and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: close below S1 with high volume confirmation
            elif current_close < s1_val and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to pivot point
            if current_close < pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to pivot point
            if current_close > pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 06:49
