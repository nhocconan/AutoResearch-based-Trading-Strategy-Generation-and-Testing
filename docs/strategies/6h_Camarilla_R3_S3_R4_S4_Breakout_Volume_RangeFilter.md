# Strategy: 6h_Camarilla_R3_S3_R4_S4_Breakout_Volume_RangeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.478 | -11.8% | -37.4% | 29 | FAIL |
| ETHUSDT | 0.090 | +21.8% | -24.3% | 19 | PASS |
| SOLUSDT | 0.981 | +221.7% | -40.3% | 12 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 1.649 | +50.6% | -7.8% | 6 | PASS |
| SOLUSDT | -1.003 | -19.6% | -39.9% | 10 | FAIL |

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
    
    # Get daily data for pivot levels (1d is HTF for 6h)
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
    r3 = daily_high + 2 * (pivot - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot)
    r4 = daily_high + 3 * (pivot - daily_low)
    s4 = daily_low - 3 * (daily_high - pivot)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, daily, r1)
    s1_aligned = align_htf_to_ltf(prices, daily, s1)
    r2_aligned = align_htf_to_ltf(prices, daily, r2)
    s2_aligned = align_htf_to_ltf(prices, daily, s2)
    r3_aligned = align_htf_to_ltf(prices, daily, r3)
    s3_aligned = align_htf_to_ltf(prices, daily, s3)
    r4_aligned = align_htf_to_ltf(prices, daily, r4)
    s4_aligned = align_htf_to_ltf(prices, daily, s4)
    
    # Volume filter: current 6h volume > 1.5x 20-period average volume
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
            # Long conditions: price breaks above R4 with volume (strong breakout)
            if close[i] > r4_aligned[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from S3 with volume (above S3, below S4)
            elif close[i] > s3_aligned[i] and close[i] < s4_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below S4 with volume (strong breakout)
            elif close[i] < s4_aligned[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at R3 with volume (below R3, above R4)
            elif close[i] < r3_aligned[i] and close[i] > r4_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Camarilla_R3_S3_R4_S4_Breakout_Volume_RangeFilter"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-15 08:43
