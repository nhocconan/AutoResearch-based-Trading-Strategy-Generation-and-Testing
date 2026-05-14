# Strategy: 6h_1d_Pivot_R2S2_MomentumBreakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.081 | +16.3% | -12.3% | 258 | FAIL |
| ETHUSDT | 0.157 | +27.9% | -14.0% | 254 | PASS |
| SOLUSDT | 0.607 | +78.6% | -16.2% | 221 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.412 | +30.6% | -6.2% | 86 | PASS |
| SOLUSDT | -0.346 | -0.2% | -17.4% | 81 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_1d_Pivot_R2S2_MomentumBreakout
# Hypothesis: Trade momentum breakouts from 1d R2/S2 levels on 6h timeframe with volume confirmation.
# R2/S2 levels provide balanced breakout points that capture momentum while filtering noise.
# Uses volume spike confirmation to ensure institutional participation.
# Works in both bull and bear markets by trading breakouts in direction of momentum.
# Targets 20-40 trades per year by requiring strong momentum breaks with volume confirmation.

name = "6h_1d_Pivot_R2S2_MomentumBreakout"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d R2 and S2 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R2 and S2 (momentum breakout levels)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    
    # Align 1d levels to 6h timeframe
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R2 with volume spike
            if (close[i] > r2_aligned[i] * 1.002 and 
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S2 with volume spike
            elif (close[i] < s2_aligned[i] * 0.998 and 
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S2 or momentum loss
            if close[i] < s2_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R2 or momentum loss
            if close[i] > r2_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 03:41
