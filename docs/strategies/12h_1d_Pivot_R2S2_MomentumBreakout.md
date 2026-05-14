# Strategy: 12h_1d_Pivot_R2S2_MomentumBreakout

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.091 | +17.5% | -6.5% | 88 | FAIL |
| ETHUSDT | 0.119 | +25.4% | -8.8% | 78 | PASS |
| SOLUSDT | -0.051 | +12.8% | -25.0% | 79 | FAIL |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.120 | +7.2% | -6.2% | 30 | PASS |

## Code
```python
#!/usr/bin/env python3
# 12h_1d_Pivot_R2S2_MomentumBreakout
# Hypothesis: Trade breakouts from 1d Camarilla R2/S2 levels on 12h timeframe with volume confirmation.
# R2/S2 levels represent stronger intraday support/resistance than R1/S1, reducing false breakouts.
# Uses volume surge (>2x 20-period average) to confirm institutional participation.
# Works in bull markets (breakouts continue) and bear markets (mean reversion from extreme levels).
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

name = "12h_1d_Pivot_R2S2_MomentumBreakout"
timeframe = "12h"
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
    
    # Calculate 1d pivot and Camarilla R2/S2 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla R2 and S2 levels (stronger intraday support/resistance)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    
    # Align 1d levels to 12h timeframe
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
            # Long: price above R2 with volume surge
            if (close[i] > r2_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S2 with volume surge
            elif (close[i] < s2_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S2
            if close[i] < s2_aligned[i] * 0.997:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R2
            if close[i] > r2_aligned[i] * 1.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 03:57
