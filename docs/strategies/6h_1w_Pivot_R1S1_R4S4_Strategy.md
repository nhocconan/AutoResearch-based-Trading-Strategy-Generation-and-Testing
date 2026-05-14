# Strategy: 6h_1w_Pivot_R1S1_R4S4_Strategy

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.446 | -6.7% | -30.1% | 260 | FAIL |
| ETHUSDT | 0.040 | +19.4% | -31.1% | 178 | PASS |
| SOLUSDT | 1.247 | +259.7% | -24.8% | 274 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.154 | +7.8% | -13.8% | 188 | PASS |
| SOLUSDT | -0.855 | -11.1% | -18.9% | 153 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-week pivot levels (R1/S1, R2/S2, R3/S3, R4/S4) 
# and volume confirmation. Fade at extreme pivots (R4/S4) with reversal signals,
# continue trend at middle pivots (R1/S1, R2/S2). Uses weekly pivot points 
# calculated from prior week's OHLC. Works in both bull and bear markets by 
# fading extremes and catching continuations. Target: 50-150 total trades over 4 years.
name = "6h_1w_Pivot_R1S1_R4S4_Strategy"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support and resistance levels
    s1_1w = (2 * pivot_1w) - high_1w
    r1_1w = (2 * pivot_1w) - low_1w
    s2_1w = pivot_1w - (high_1w - low_1w)
    r2_1w = pivot_1w + (high_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    
    # Volume filter: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Fade at extreme pivots (R4/S4) - mean reversion
            if close[i] <= s4_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] >= r4_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Continue trend at middle pivots (R1/S1, R2/S2) - breakout
            elif close[i] > r2_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < s2_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit at R3 (take profit) or S4 (stop reversal)
            if close[i] >= r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] <= s4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit at S3 (take profit) or R4 (stop reversal)
            if close[i] <= s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] >= r4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-19 17:16
