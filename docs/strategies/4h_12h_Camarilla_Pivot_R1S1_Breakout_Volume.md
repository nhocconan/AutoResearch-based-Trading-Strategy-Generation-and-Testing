# Strategy: 4h_12h_Camarilla_Pivot_R1S1_Breakout_Volume

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.249 | +31.5% | -8.4% | 437 | PASS |
| ETHUSDT | 0.171 | +28.5% | -7.7% | 429 | PASS |
| SOLUSDT | 0.078 | +21.7% | -18.0% | 387 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.512 | -7.1% | -13.2% | 162 | FAIL |
| ETHUSDT | 0.759 | +17.1% | -7.2% | 155 | PASS |
| SOLUSDT | 0.476 | +12.7% | -10.2% | 136 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_12h_Camarilla_Pivot_R1S1_Breakout_Volume
# Hypothesis: Trade breakouts from 12h Camarilla R1/S1 levels on 4h timeframe with volume confirmation.
# Uses 12h pivot levels for institutional reference points, volume surge for confirmation.
# Designed for 20-50 trades per year by requiring precise level breaks with volume surge.
# Works in bull markets (breakouts continue) and bear markets (mean reversion from extreme levels).

name = "4h_12h_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h pivot and Camarilla R1/S1 levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla R1 and S1 levels (primary intraday support/resistance)
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    
    # Align 12h levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R1 with volume surge
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 1.8 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S1 with volume surge
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 1.8 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1
            if close[i] < s1_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1
            if close[i] > r1_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 03:53
