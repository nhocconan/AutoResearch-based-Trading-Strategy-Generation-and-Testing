#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: In trending markets, breakouts from Donchian channels aligned with weekly pivot direction capture sustained moves. Volume confirms institutional participation. Works in bull (long with bullish weekly pivot) and bear (short with bearish weekly pivot).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (P, R1, S1, R2, S2, R3, S3, R4, S4)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot = (high_weekly + low_weekly + close_weekly) / 3
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    r4 = 3 * pivot + (high_weekly - 3 * low_weekly)
    s4 = 3 * pivot - (3 * high_weekly - low_weekly)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * (donchian_high[i] - donchian_low[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * (donchian_high[i] - donchian_low[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            # Long: break above Donchian high with bullish weekly pivot (price > weekly R3)
            long_setup = (close[i] > donchian_high[i] and 
                         close[i] > r4_aligned[i] and 
                         vol_filter[i])
            # Short: break below Donchian low with bearish weekly pivot (price < weekly S3)
            short_setup = (close[i] < donchian_low[i] and 
                          close[i] < s4_aligned[i] and 
                          vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: In trending markets, breakouts from Donchian channels aligned with weekly pivot direction capture sustained moves. Volume confirms institutional participation. Works in bull (long with bullish weekly pivot) and bear (short with bearish weekly pivot).
Target: 75-150 total trades over 4 years (19-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (P, R1, S1, R2, S2, R3, S3, R4, S4)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    pivot = (high_weekly + low_weekly + close_weekly) / 3
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot)
    r4 = 3 * pivot + (high_weekly - 3 * low_weekly)
    s4 = 3 * pivot - (3 * high_weekly - low_weekly)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or
                close[i] <= entry_price - 2.5 * (donchian_high[i] - donchian_low[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or
                close[i] >= entry_price + 2.5 * (donchian_high[i] - donchian_low[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            # Long: break above Donchian high with bullish weekly pivot (price > weekly R3)
            long_setup = (close[i] > donchian_high[i] and 
                         close[i] > r4_aligned[i] and 
                         vol_filter[i])
            # Short: break below Donchian low with bearish weekly pivot (price < weekly S3)
            short_setup = (close[i] < donchian_low[i] and 
                          close[i] < s4_aligned[i] and 
                          vol_filter[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>