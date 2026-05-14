# Strategy: 6h_1d_WeeklyPivot_Donchian_Volume_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.247 | +31.6% | -7.3% | 113 | PASS |
| ETHUSDT | 0.680 | +63.9% | -9.8% | 96 | PASS |
| SOLUSDT | 0.641 | +85.2% | -24.5% | 84 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -1.390 | -7.9% | -11.6% | 44 | FAIL |
| ETHUSDT | 0.462 | +12.9% | -6.6% | 36 | PASS |
| SOLUSDT | -0.590 | -4.1% | -16.1% | 30 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d weekly pivot context and volume confirmation
# Weekly pivot levels (from 1d data) provide institutional support/resistance
# Donchian breakout captures momentum in direction of pivot bias
# Volume > 2x average confirms institutional participation
# Works in bull/bear as pivot bias adapts to trend
# Target: 15-25 trades/year per symbol (60-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for weekly pivot and trend context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week (using 1d data)
    # Weekly high/low/close from previous 5 trading days
    lookback = 5
    if len(df_1d) < lookback:
        return np.zeros(n)
    
    # Get prior week's OHLC (excluding current incomplete day)
    prev_week_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    prev_week_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    prev_week_close = pd.Series(df_1d['close']).rolling(window=lookback, min_periods=lookback).last().shift(1).values
    
    # Weekly pivot calculation (standard floor trader pivot)
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = 2 * weekly_pivot - prev_week_low
    weekly_s1 = 2 * weekly_pivot - prev_week_high
    weekly_r2 = weekly_pivot + (prev_week_high - prev_week_low)
    weekly_s2 = weekly_pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Donchian channel (20 periods) on 6h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Pivot bias: price relative to weekly pivot
        above_pivot = close[i] > weekly_pivot_aligned[i]
        below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Volume confirmation: current volume > 2x average
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + above weekly pivot + volume
            if (close[i] > dc_upper[i] and 
                above_pivot and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + below weekly pivot + volume
            elif (close[i] < dc_lower[i] and 
                  below_pivot and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly pivot or breaks below S1
            if close[i] < weekly_pivot_aligned[i] or close[i] < weekly_s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly pivot or breaks above R1
            if close[i] > weekly_pivot_aligned[i] or close[i] > weekly_r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_WeeklyPivot_Donchian_Volume_v1"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-14 06:56
