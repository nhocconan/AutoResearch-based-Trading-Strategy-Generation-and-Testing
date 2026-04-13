#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly Donchian breakout + volume confirmation + volume surge filter
# Strategy: Long when price breaks above weekly Donchian high (20) with volume > 2x 20-day average
# Short when price breaks below weekly Donchian low (20) with volume > 2x 20-day average
# Weekly Donchian provides strong structural levels; volume surge confirms institutional interest
# Target: 20-30 total trades over 4 years (5-7.5/year) to minimize fee drag and capture major trends

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(high_20_1w_aligned[i]) or 
            np.isnan(low_20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition (2x average)
        volume_surge = volume[i] > 2.0 * vol_ma_20[i]
        
        # Breakout conditions
        long_breakout = close[i] > high_20_1w_aligned[i]
        short_breakout = close[i] < low_20_1w_aligned[i]
        
        # Entry logic
        long_entry = long_breakout and volume_surge
        short_entry = short_breakout and volume_surge
        
        # Exit conditions: opposite breakout
        exit_long = position == 1 and short_breakout
        exit_short = position == -1 and long_breakout
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "1d_1w_donchian_breakout_volume_surge_v1"
timeframe = "1d"
leverage = 1.0