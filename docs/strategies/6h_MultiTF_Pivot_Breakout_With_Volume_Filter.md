# Strategy: 6h_MultiTF_Pivot_Breakout_With_Volume_Filter

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.087 | +17.4% | -11.0% | 319 | FAIL |
| ETHUSDT | 0.298 | +34.1% | -9.3% | 307 | PASS |
| SOLUSDT | 0.732 | +84.4% | -15.4% | 261 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| ETHUSDT | 0.862 | +17.6% | -6.2% | 104 | PASS |
| SOLUSDT | -0.041 | +4.9% | -10.4% | 102 | FAIL |

## Code
```python
#!/usr/bin/env python3
"""
6h_MultiTF_Pivot_Breakout_With_Volume_Filter
Hypothesis: Trade 6h price breakouts above/below 12h pivot resistance/support levels with volume confirmation and 1d trend filter.
Long when price breaks above 12h R1 with volume spike and 1d uptrend; short when breaks below 12h S1 with volume spike and 1d downtrend.
Uses 12h pivot levels (calculated from prior 12h bar) and volume > 1.5x 20-period average for confirmation.
Designed for 6h timeframe to capture medium-term moves while reducing noise.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, volume filter reduces false breakouts.
"""

name = "6h_MultiTF_Pivot_Breakout_With_Volume_Filter"
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
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h pivot points (using prior 12h bar's high, low, close)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point calculation: PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    pp_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = 2 * pp_12h - low_12h
    s1_12h = 2 * pp_12h - high_12h
    
    # Align 12h pivot levels to 6h timeframe (already delayed by one bar via align_htf_to_ltf)
    pp_12h_aligned = align_htf_to_ltf(prices, df_12h, pp_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA20 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema20_1d = ema(close_1d, 20)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are ready (20 for EMA + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 12h R1 with volume filter AND 1d uptrend (close > EMA20)
            if close[i] > r1_12h_aligned[i] and volume_filter[i] and close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h S1 with volume filter AND 1d downtrend (close < EMA20)
            elif close[i] < s1_12h_aligned[i] and volume_filter[i] and close[i] < ema20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 12h pivot point OR 1d trend turns down
            if close[i] < pp_12h_aligned[i] or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 12h pivot point OR 1d trend turns up
            if close[i] > pp_12h_aligned[i] or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
```

## Last Updated
2026-04-20 04:39
