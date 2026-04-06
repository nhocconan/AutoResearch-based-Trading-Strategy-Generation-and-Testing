#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Donchian(20) breakouts capture institutional breakout moves. 
Weekly pivot (calculated from prior week) provides directional bias: 
only long when price above weekly pivot, short when below. 
Volume confirms institutional participation. Works in bull (breakouts up) 
and bear (breakdowns down). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    weekly_pivot = (high_w + low_w + close_w) / 3.0
    weekly_pivot_prev = np.roll(weekly_pivot, 1)
    weekly_pivot_prev[0] = weekly_pivot[0]  # first value
    weekly_bullish = weekly_pivot > weekly_pivot_prev  # pivot rising = bullish bias
    weekly_bearish = weekly_pivot < weekly_pivot_prev   # pivot falling = bearish bias
    
    # Align weekly data to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bearish)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channels
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse Donchian break or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + weekly pivot bias + volume
            bull_break = close[i] > donchian_high[i]
            bear_break = close[i] < donchian_low[i]
            high_volume = volume[i] > vol_ema[i] * 2.0
            
            if bull_break and weekly_bullish_aligned[i] and high_volume:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_break and weekly_bearish_aligned[i] and high_volume:
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
Hypothesis: Donchian(20) breakouts capture institutional breakout moves. 
Weekly pivot (calculated from prior week) provides directional bias: 
only long when price above weekly pivot, short when below. 
Volume confirms institutional participation. Works in bull (breakouts up) 
and bear (breakdowns down). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    weekly_pivot = (high_w + low_w + close_w) / 3.0
    weekly_pivot_prev = np.roll(weekly_pivot, 1)
    weekly_pivot_prev[0] = weekly_pivot[0]  # first value
    weekly_bullish = weekly_pivot > weekly_pivot_prev  # pivot rising = bullish bias
    weekly_bearish = weekly_pivot < weekly_pivot_prev   # pivot falling = bearish bias
    
    # Align weekly data to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bearish)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channels
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse Donchian break or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + weekly pivot bias + volume
            bull_break = close[i] > donchian_high[i]
            bear_break = close[i] < donchian_low[i]
            high_volume = volume[i] > vol_ema[i] * 2.0
            
            if bull_break and weekly_bullish_aligned[i] and high_volume:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_break and weekly_bearish_aligned[i] and high_volume:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals