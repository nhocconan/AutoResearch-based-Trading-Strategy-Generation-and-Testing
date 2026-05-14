# Strategy: 12h_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.329 | +42.1% | -17.0% | 87 | PASS |
| ETHUSDT | 0.876 | +120.8% | -19.7% | 81 | PASS |
| SOLUSDT | 0.908 | +194.0% | -29.7% | 76 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.188 | -10.8% | -15.6% | 43 | FAIL |
| ETHUSDT | 0.014 | +4.3% | -19.7% | 35 | PASS |
| SOLUSDT | 0.851 | +29.0% | -18.9% | 26 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels (1d is HTF for 12h)
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily pivot levels (classic floor trader pivots)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    r2_aligned = align_htf_to_ltf(prices, daily, r2)
    s2_aligned = align_htf_to_ltf(prices, daily, s2)
    
    # Volume filter: current 12h volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Range filter: avoid trading when price is within 0.5% of pivot
    price_to_pivot = np.abs(close - pivot_aligned) / pivot_aligned
    range_filter = price_to_pivot > 0.005
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter and range filter both pass
        if volume_filter[i] and range_filter[i]:
            # Long conditions: price breaks above R1 with volume
            if close[i] > r1_aligned[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from S1 with volume (above S1, below S2)
            elif close[i] > s1_aligned[i] and close[i] < s2_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S1 with volume
            elif close[i] < s1_aligned[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at R1 with volume (below R1, above R2)
            elif close[i] < r1_aligned[i] and close[i] > r2_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-15 08:42
