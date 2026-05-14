# Strategy: 4h_DailyPivot_R2_S2_Breakout_Volume_RangeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.342 | +43.6% | -16.0% | 109 | PASS |
| ETHUSDT | 0.049 | +17.7% | -22.5% | 94 | PASS |
| SOLUSDT | 0.266 | +40.2% | -41.2% | 99 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.525 | -14.5% | -23.4% | 41 | FAIL |
| ETHUSDT | 0.448 | +14.9% | -16.9% | 37 | PASS |
| SOLUSDT | 0.477 | +16.6% | -14.2% | 31 | PASS |

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
    
    # Get daily data for 4h context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    daily_volume = daily['volume'].values
    
    # Calculate daily pivot levels (standard pivot point)
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_r1 = 2 * daily_pivot - daily_low
    daily_s1 = 2 * daily_pivot - daily_high
    daily_r2 = daily_pivot + (daily_high - daily_low)
    daily_s2 = daily_pivot - (daily_high - daily_low)
    
    # Align daily pivot levels to 4h timeframe
    daily_pivot_4h = align_htf_to_ltf(prices, daily, daily_pivot)
    daily_r1_4h = align_htf_to_ltf(prices, daily, daily_r1)
    daily_s1_4h = align_htf_to_ltf(prices, daily, daily_s1)
    daily_r2_4h = align_htf_to_ltf(prices, daily, daily_r2)
    daily_s2_4h = align_htf_to_ltf(prices, daily, daily_s2)
    
    # Volume filter: current 4h volume > 1.5x 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Range filter: avoid trading when price is within 0.3% of daily pivot
    price_to_pivot = np.abs(close - daily_pivot_4h) / daily_pivot_4h
    range_filter = price_to_pivot > 0.003
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_pivot_4h[i]) or np.isnan(daily_r1_4h[i]) or 
            np.isnan(daily_s1_4h[i]) or np.isnan(daily_r2_4h[i]) or 
            np.isnan(daily_s2_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when both filters pass
        if volume_filter[i] and range_filter[i]:
            # Long conditions: price breaks above daily R2 with volume
            if close[i] > daily_r2_4h[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below daily S2 with volume
            elif close[i] < daily_s2_4h[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_DailyPivot_R2_S2_Breakout_Volume_RangeFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-15 08:53
