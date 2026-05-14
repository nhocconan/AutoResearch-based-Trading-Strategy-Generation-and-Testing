# Strategy: 4h_RangeBreakout_WeeklyPivot_VolumeSpike_v2

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.084 | +17.4% | -14.2% | 100 | FAIL |
| ETHUSDT | 0.450 | +44.1% | -8.8% | 81 | PASS |
| SOLUSDT | 0.579 | +70.0% | -12.7% | 84 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.370 | +10.8% | -5.9% | 33 | PASS |
| SOLUSDT | 0.270 | +9.4% | -8.5% | 25 | PASS |

## Code
```python
#!/usr/bin/env python3
name = "4h_RangeBreakout_WeeklyPivot_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    week_high = df_1d['high'].rolling(window=7, min_periods=7).max().shift(1).values
    week_low = df_1d['low'].rolling(window=7, min_periods=7).min().shift(1).values
    week_close = df_1d['close'].shift(1).values
    
    # Classic pivot points for weekly range
    weekly_pivot = (week_high + week_low + week_close) / 3
    r1 = 2 * weekly_pivot - week_low
    s1 = 2 * weekly_pivot - week_high
    
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h range breakout detection (24h lookback = 6 periods)
    range_high = pd.Series(high).rolling(window=6, min_periods=6).max().shift(1).values  # Previous 6 periods (24h)
    range_low = pd.Series(low).rolling(window=6, min_periods=6).min().shift(1).values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(6, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 AND above 4h range high + volume spike
            if close[i] > r1_aligned[i] and close[i] > range_high[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 AND below 4h range low + volume spike
            elif close[i] < s1_aligned[i] and close[i] < range_low[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to weekly pivot or range breaks in opposite direction
            if position == 1:
                if close[i] < weekly_pivot_aligned[i] or close[i] < range_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_pivot_aligned[i] or close[i] > range_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-05-07 05:31
