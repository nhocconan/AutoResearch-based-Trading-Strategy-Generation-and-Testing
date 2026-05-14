# Strategy: 12h_DailyPivot_Breakout_Volume_TrendFilter_Strict

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.199 | +29.3% | -8.2% | 71 | PASS |
| ETHUSDT | 0.022 | +20.2% | -12.8% | 61 | PASS |
| SOLUSDT | 0.682 | +90.6% | -26.3% | 57 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.869 | -2.1% | -9.6% | 26 | FAIL |
| ETHUSDT | 0.551 | +14.3% | -6.8% | 24 | PASS |
| SOLUSDT | -1.324 | -13.8% | -22.1% | 24 | FAIL |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 2.0 * 24-period average (12h bars)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Choppiness index filter (trending market filter)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(volume_ma24[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma24[i])
        
        # Choppiness filter: only trade in trending markets (CHOP < 38.2)
        trend_filter = chop[i] < 38.2
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and trend filter
            if close[i] > r1_12h[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and trend filter
            elif close[i] < s1_12h[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1
            if close[i] < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1
            if close[i] > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_Volume_TrendFilter_Strict"
timeframe = "12h"
leverage = 1.0
```

## Last Updated
2026-04-17 13:10
