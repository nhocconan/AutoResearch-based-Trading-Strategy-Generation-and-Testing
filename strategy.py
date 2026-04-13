#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation (trend filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Get 1d data for pivot calculation (entry zones)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day data)
    pivot = np.full(len(high_1d), np.nan)
    r1 = np.full(len(high_1d), np.nan)
    s1 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        pivot[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = 2 * pivot[i] - low_1d[i-1]
        s1[i] = 2 * pivot[i] - high_1d[i-1]
    
    # Align all indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average volume (6h)
        vol_ma = np.mean(volume[max(0, i-20):i+1]) if i >= 20 else np.mean(volume[:i+1])
        vol_confirm = volume[i] > 1.5 * vol_ma
        
        # Donchian breakout conditions (trend filter)
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # Pivot bounce conditions (entry at support/resistance)
        pivot_support = close[i] > s1_aligned[i] and close[i] < pivot_aligned[i]
        pivot_resistance = close[i] < r1_aligned[i] and close[i] > pivot_aligned[i]
        
        # Entry conditions with confluence
        long_entry = donchian_breakout_long and vol_confirm and pivot_support
        short_entry = donchian_breakout_short and vol_confirm and pivot_resistance
        
        # Exit conditions: opposite Donchian breakout or pivot reversal
        exit_long = position == 1 and (donchian_breakout_short or close[i] > r1_aligned[i])
        exit_short = position == -1 and (donchian_breakout_long or close[i] < s1_aligned[i])
        
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

name = "6h_12h_1d_donchian_pivot_bounce"
timeframe = "6h"
leverage = 1.0