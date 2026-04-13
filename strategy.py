#!/usr/bin/env python3
"""
12h_1w_Donchian_Breakout_With_Volume_Confirmation
Hypothesis: Combines weekly Donchian channel breakout with volume confirmation on 12h timeframe.
In trending markets, price breaks out of weekly Donchian(20) channel with above-average volume.
Works in both bull and bear markets by capturing strong directional moves after consolidation.
Target: 12-37 trades/year on 12h (50-150 total over 4 years).
"""

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
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20 periods)
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    
    # Weekly breakout conditions
    breakout_up = close_1w > highest_high_20
    breakout_down = close_1w < lowest_low_20
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    volume_expansion_1d = volume_1d > (vol_ma_20_1d * 1.5)
    
    # Align all signals to 12h timeframe
    breakout_up_aligned = align_htf_to_ltf(prices, df_1w, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1w, breakout_down)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(breakout_up_aligned[i]) or \
           np.isnan(breakout_down_aligned[i]) or \
           np.isnan(volume_expansion_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: weekly breakout with volume expansion
        if breakout_up_aligned[i] and volume_expansion_aligned[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif breakout_down_aligned[i] and volume_expansion_aligned[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        elif position == 1:
            # Hold long position
            signals[i] = position_size
        elif position == -1:
            # Hold short position
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1w_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0