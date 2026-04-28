#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR filter.
# Enter long when price breaks above 1w Donchian upper channel (20-period high) and volume > 1.5x 20-bar average.
# Enter short when price breaks below 1w Donchian lower channel (20-period low) and volume > 1.5x 20-bar average.
# Exit when price returns to the 1w Donchian midpoint or opposite breakout occurs.
# Donchian channels provide clear structure, breakouts capture momentum, volume confirms validity.
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
# Uses discrete position sizing (0.25) to control risk. Target: 30-100 total trades over 4 years.

name = "1d_Donchian20_1w_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w Donchian components (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Upper channel: 20-period high
    upper_channel = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    lower_channel = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Midpoint: (upper + lower) / 2
    midpoint = (upper_channel + lower_channel) / 2.0
    
    # Align Donchian components to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
    
    # Calculate 1d volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        long_breakout = close[i] > upper_aligned[i]
        short_breakout = close[i] < lower_aligned[i]
        
        # Exit conditions: price returns to midpoint or opposite breakout
        long_exit = close[i] < midpoint_aligned[i]
        short_exit = close[i] > midpoint_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and volume_confirm[i]
        short_entry = short_breakout and volume_confirm[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals