# Strategy: 1d_WeeklyPivot_R1_S1_Breakout_Volume_RangeFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.076 | +22.5% | -21.4% | 12 | PASS |
| ETHUSDT | -0.170 | +1.8% | -26.1% | 13 | FAIL |
| SOLUSDT | 0.605 | +104.1% | -27.4% | 14 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.942 | +20.1% | -6.1% | 4 | PASS |
| SOLUSDT | -1.441 | -28.8% | -42.3% | 7 | FAIL |

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
    
    # Get daily data for weekly pivot levels
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate weekly pivot levels from daily data (last 5 days)
    weekly_high = pd.Series(daily_high).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(daily_low).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(daily_close).rolling(window=5, min_periods=5).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, daily, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, daily, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, daily, weekly_s1)
    
    # Volume filter: current daily volume > 1.8x 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    # Range filter: avoid trading when price is within 0.5% of weekly pivot
    price_to_pivot = np.abs(close - weekly_pivot_aligned) / weekly_pivot_aligned
    range_filter = price_to_pivot > 0.005
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when both filters pass
        if volume_filter[i] and range_filter[i]:
            # Long conditions: price breaks above weekly R1 with volume
            if close[i] > weekly_r1_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below weekly S1 with volume
            elif close[i] < weekly_s1_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_WeeklyPivot_R1_S1_Breakout_Volume_RangeFilter"
timeframe = "1d"
leverage = 1.0
```

## Last Updated
2026-04-15 08:51
