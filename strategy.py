#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Pivot + Volume
Hypothesis: Combining 6h Donchian breakouts with daily pivot levels and volume confirmation
creates high-probability entries. Pivot levels act as institutional support/resistance,
while volume confirms breakout strength. Works in both bull and bear markets by trading
breakouts in direction of higher timeframe bias.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 (R1) = (2 * PP) - L
    r1 = (2 * pp) - low_1d
    # Support 1 (S1) = (2 * PP) - H
    s1 = (2 * pp) - high_1d
    # Resistance 2 (R2) = PP + (H - L)
    r2 = pp + (high_1d - low_1d)
    # Support 2 (S2) = PP - (H - L)
    s2 = pp - (high_1d - low_1d)
    # Resistance 3 (R3) = H + 2*(PP - L)
    r3 = high_1d + 2 * (pp - low_1d)
    # Support 3 (S3) = L - 2*(H - PP)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for completed bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(pp_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below S1
            if close[i] < lowest_low or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above R1
            if close[i] > highest_high or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + pivot filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Additional filter: breakout should be beyond S1/R1 for strength
            strong_bull = bull_breakout and close[i] > s1_aligned[i]
            strong_bear = bear_breakout and close[i] < r1_aligned[i]
            
            if strong_bull and volume_filter:
                signals[i] = 0.25
                position = 1
            elif strong_bear and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Pivot + Volume
Hypothesis: Combining 6h Donchian breakouts with daily pivot levels and volume confirmation
creates high-probability entries. Pivot levels act as institutional support/resistance,
while volume confirms breakout strength. Works in both bull and bear markets by trading
breakouts in direction of higher timeframe bias.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 (R1) = (2 * PP) - L
    r1 = (2 * pp) - low_1d
    # Support 1 (S1) = (2 * PP) - H
    s1 = (2 * pp) - high_1d
    # Resistance 2 (R2) = PP + (H - L)
    r2 = pp + (high_1d - low_1d)
    # Support 2 (S2) = PP - (H - L)
    s2 = pp - (high_1d - low_1d)
    # Resistance 3 (R3) = H + 2*(PP - L)
    r3 = high_1d + 2 * (pp - low_1d)
    # Support 3 (S3) = L - 2*(H - PP)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align pivot levels to 6h timeframe (shifted by 1 day for completed bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if pivot data not available
        if np.isnan(pp_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below S1
            if close[i] < lowest_low or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above R1
            if close[i] > highest_high or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + pivot filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Additional filter: breakout should be beyond S1/R1 for strength
            strong_bull = bull_breakout and close[i] > s1_aligned[i]
            strong_bear = bear_breakout and close[i] < r1_aligned[i]
            
            if strong_bull and volume_filter:
                signals[i] = 0.25
                position = 1
            elif strong_bear and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals