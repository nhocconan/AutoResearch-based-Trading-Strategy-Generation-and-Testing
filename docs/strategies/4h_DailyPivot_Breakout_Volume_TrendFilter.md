# Strategy: 4h_DailyPivot_Breakout_Volume_TrendFilter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.013 | +18.1% | -11.8% | 132 | PASS |
| ETHUSDT | 0.011 | +15.3% | -25.5% | 119 | PASS |
| SOLUSDT | 0.451 | +73.0% | -36.3% | 123 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.452 | -0.5% | -10.9% | 43 | FAIL |
| ETHUSDT | 0.588 | +17.7% | -10.7% | 41 | PASS |
| SOLUSDT | 0.211 | +9.1% | -15.5% | 42 | PASS |

## Code
```python
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    
    # Align daily pivot levels to 4h timeframe (use previous day's levels)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_4h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_4h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness index filter (trending market filter)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    chop = 100 * np.log10((highest_high - lowest_low) / (atr_safe * 14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or
            np.isnan(r2_4h[i]) or np.isnan(s2_4h[i]) or np.isnan(volume_ma20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Choppiness filter: only trade in trending markets (CHOP < 38.2)
        trend_filter = chop[i] < 38.2
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and trend filter
            if close[i] > r1_4h[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and trend filter
            elif close[i] < s1_4h[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S2 (deeper level) OR chop increases (range developing)
            if close[i] < s2_4h[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R2 OR chop increases (range developing)
            if close[i] > r2_4h[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyPivot_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0
```

## Last Updated
2026-04-17 12:39
