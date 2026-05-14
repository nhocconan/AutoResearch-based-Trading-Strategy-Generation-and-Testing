# Strategy: 6h_DailyPivot_Breakout_Volume_TrendFilter_Strict

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | 0.079 | +23.5% | -10.1% | 137 | PASS |
| ETHUSDT | 0.327 | +40.1% | -14.1% | 125 | PASS |
| SOLUSDT | 0.938 | +142.5% | -20.9% | 95 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -2.050 | -14.3% | -15.8% | 58 | FAIL |
| ETHUSDT | 1.138 | +26.0% | -6.6% | 38 | PASS |
| SOLUSDT | -0.370 | -1.4% | -17.7% | 37 | FAIL |

## Code
```python
#!/usr/bin/env python3
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
    
    # Get daily data for pivot points (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 6h timeframe (use previous day's levels)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 2.0 * 30-period average (balanced)
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Choppiness index filter (trending market filter)
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
    
    start_idx = 30  # Need sufficient data for volume MA and chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(volume_ma30[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (2.0 * volume_ma30[i])
        
        # Choppiness filter: only trade in trending markets (CHOP < 38.2)
        trend_filter = chop[i] < 38.2
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and trend filter
            if close[i] > r1_6h[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume and trend filter
            elif close[i] < s1_6h[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1
            if close[i] < s1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1
            if close[i] > r1_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_Breakout_Volume_TrendFilter_Strict"
timeframe = "6h"
leverage = 1.0
```

## Last Updated
2026-04-17 12:58
