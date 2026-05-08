#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Daily Donchian channel (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot direction: 1 if weekly close > previous weekly close, else -1
    weekly_direction = np.where(close_1w > np.roll(close_1w, 1), 1, -1)
    weekly_direction[0] = 1  # initialize first value
    
    # Align daily Donchian and weekly direction to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    weekly_dir_aligned = align_htf_to_ltf(prices, df_1w, weekly_direction)
    
    # Volume filter: current 1d volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_strong = volume_1d > (1.5 * vol_ma20)
    volume_strong_aligned = align_htf_to_ltf(prices, df_1d, volume_strong)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(weekly_dir_aligned[i]) or np.isnan(volume_strong_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, weekly trend up, strong volume
            long_cond = (close[i] > donch_high_aligned[i] and 
                        weekly_dir_aligned[i] > 0 and
                        volume_strong_aligned[i])
            
            # Short: Price breaks below Donchian low, weekly trend down, strong volume
            short_cond = (close[i] < donch_low_aligned[i] and 
                         weekly_dir_aligned[i] < 0 and
                         volume_strong_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR weekly trend turns down
            if close[i] < donch_low_aligned[i] or weekly_dir_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR weekly trend turns up
            if close[i] > donch_high_aligned[i] or weekly_dir_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals