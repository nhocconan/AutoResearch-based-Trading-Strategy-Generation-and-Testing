#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1d/1w Camarilla pivot direction + volume confirmation (>2.0x 20-period MA).
    # Uses higher timeframe pivot structure to filter breakout direction: long only when price > 1d pivot, short only when price < 1d pivot.
    # Weekly pivot confirms longer-term bias: requires alignment between 1d and 1w pivot zones.
    # Volume filter ensures institutional participation. Discrete sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 75-150 total trades over 4 years (19-38/year) to stay within fee drag limits.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for pivot calculations (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1
    s3_1d = pivot_1d - range_1d * 1.1
    r4_1d = pivot_1d + range_1d * 1.5
    s4_1d = pivot_1d - range_1d * 1.5
    
    # Calculate Camarilla pivot levels for 1w
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1
    s3_1w = pivot_1w - range_1w * 1.1
    r4_1w = pivot_1w + range_1w * 1.5
    s4_1w = pivot_1w - range_1w * 1.5
    
    # Align HTF pivot levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate 20-period volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian(20) channels on primary timeframe (6h)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0 * 20-period MA (stricter to reduce trades)
        volume_filter = volume[i] > 2.0 * volume_ma[i]
        
        # Pivot direction filters: price relative to 1d S3/R3 and 1w pivot
        # Long bias: price > 1d S3 and price > 1w pivot
        long_bias = (close[i] > s3_1d_aligned[i]) and (close[i] > pivot_1w_aligned[i])
        # Short bias: price < 1d R3 and price < 1w pivot
        short_bias = (close[i] < r3_1d_aligned[i]) and (close[i] < pivot_1w_aligned[i])
        
        # Donchian breakout conditions with pivot filters
        long_breakout = (close[i] > highest_high[i-1]) and volume_filter and long_bias
        short_breakout = (close[i] < lowest_low[i-1]) and volume_filter and short_bias
        
        # Exit conditions: price returns to midpoint of Donchian channel
        midpoint = (highest_high[i] + lowest_low[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_camarilla_pivot_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0