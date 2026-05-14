# Strategy: 6h_12h_Camarilla_R1S1_Breakout_Volume_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.441 | +12.3% | -5.1% | 114 | FAIL |
| ETHUSDT | 0.195 | +27.2% | -4.5% | 89 | PASS |
| SOLUSDT | 0.214 | +30.9% | -12.9% | 85 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.530 | +10.9% | -3.4% | 40 | PASS |
| SOLUSDT | -0.008 | +5.9% | -5.1% | 31 | FAIL |

## Code
```python
#!/usr/bin/env python3
# 6h_12h_Camarilla_R1S1_Breakout_Volume_TrendFilter
# Hypothesis: Use 12h Camarilla pivot levels (R1/S1) with 6h breakout, volume confirmation, and 12h EMA34 trend filter.
# Only trade breakouts aligned with trend. Targets 12-37 trades/year by using 6h timeframe and tight entry conditions.
# Designed to work in both bull and bear markets by following trend and requiring volume confirmation.

name = "6h_12h_Camarilla_R1S1_Breakout_Volume_TrendFilter"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    
    # Pivot point and ranges
    pivot_12h = typical_price_12h
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R1, S1
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    
    # Align 12h levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume spike and uptrend
            if (close[i] > r1_aligned[i] * 1.005 and 
                volume[i] > 2.5 * volume_ma[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume spike and downtrend
            elif (close[i] < s1_aligned[i] * 0.995 and 
                  volume[i] > 2.5 * volume_ma[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend reverses
            if close[i] < s1_aligned[i] or close[i] < ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend reverses
            if close[i] > r1_aligned[i] or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 02:07
