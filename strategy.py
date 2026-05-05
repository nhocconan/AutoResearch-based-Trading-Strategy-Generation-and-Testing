#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + daily pivot direction + volume confirmation
# Long when price breaks above 6h Donchian(20) high AND price > daily pivot point AND volume > 1.5 * avg_volume(20) on 6h
# Short when price breaks below 6h Donchian(20) low AND price < daily pivot point AND volume > 1.5 * avg_volume(20) on 6h
# Exit when price touches the opposite Donchian band (mean reversion) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Donchian breakout captures momentum in trending markets
# Daily pivot provides institutional reference for bias (above/below pivot)
# Volume confirmation ensures breakout strength and reduces false signals
# Works in bull markets (buying breakouts above pivot) and bear markets (selling breakdowns below pivot)

name = "6h_Donchian20_1dPivot_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point: (High + Low + Close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Get 6h data ONCE before loop for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate 6h Donchian(20)
    highest_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, highest_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, lowest_low_20)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above, above pivot, volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and close[i] > pivot_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below, below pivot, volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and close[i] < pivot_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches lower Donchian band OR volume drops below average
            if close[i] < donchian_low_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches upper Donchian band OR volume drops below average
            if close[i] > donchian_high_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals