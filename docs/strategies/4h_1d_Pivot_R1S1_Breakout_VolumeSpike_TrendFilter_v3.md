# Strategy: 4h_1d_Pivot_R1S1_Breakout_VolumeSpike_TrendFilter_v3

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.045 | +19.5% | -7.3% | 186 | FAIL |
| ETHUSDT | 0.358 | +35.9% | -8.8% | 156 | PASS |
| SOLUSDT | 0.702 | +72.5% | -15.0% | 134 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.759 | +14.5% | -4.4% | 60 | PASS |
| SOLUSDT | 0.517 | +11.5% | -7.7% | 44 | PASS |

## Code
```python
#!/usr/bin/env python3
# 4h_1d_Pivot_R1S1_Breakout_VolumeSpike_TrendFilter_v3
# Hypothesis: Use 1d Camarilla pivot levels (R1/S1) with 4h breakout, volume confirmation, and 4h EMA34 trend filter.
# Only trade breakouts aligned with trend. Tighten volume threshold to 2.5x average and increase breakout filter to 0.5% above/below pivot to reduce trades and improve quality.
# Target: 15-30 trades/year per symbol for low fee attrition and strong edge in both bull and bear markets.

name = "4h_1d_Pivot_R1S1_Breakout_VolumeSpike_TrendFilter_v3"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    
    # Pivot point and ranges
    pivot_1d = typical_price_1d
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    
    # Align 1d levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 4h EMA34 for trend filter
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume spike and uptrend
            if (close[i] > r1_aligned[i] * 1.005 and 
                volume[i] > 2.5 * volume_ma[i] and 
                close[i] > ema34[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume spike and downtrend
            elif (close[i] < s1_aligned[i] * 0.995 and 
                  volume[i] > 2.5 * volume_ma[i] and 
                  close[i] < ema34[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend reverses
            if close[i] < s1_aligned[i] or close[i] < ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend reverses
            if close[i] > r1_aligned[i] or close[i] > ema34[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 02:06
