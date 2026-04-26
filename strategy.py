#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_v1
Hypothesis: Combine weekly pivot direction (long above weekly PP, short below) with 6h Donchian(20) breakout and volume confirmation (>1.5x average volume). Weekly pivot provides robust trend filter that works in both bull/bear markets, while Donchian breakouts capture momentum. Volume confirmation avoids false breakouts. Discrete position sizing (0.25) minimizes fee churn. Target 12-37 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian, volume
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior completed weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (PP) and support/resistance levels
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    weekly_r1 = 2 * weekly_pp - low_1w
    weekly_s1 = 2 * weekly_pp - high_1w
    
    # Align weekly pivot levels to 6h timeframe (1 bar delay for completed weekly bar)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian, 20 for volume)
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        pp_val = weekly_pp_aligned[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        upper_donch = highest_high[i]
        lower_donch = lowest_low[i]
        
        # Skip if any data not ready
        if (np.isnan(pp_val) or np.isnan(avg_vol) or np.isnan(upper_donch) or 
            np.isnan(lower_donch)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price above weekly PP AND breaks above Donchian upper with volume confirmation
        long_condition = (close_val > pp_val) and (high_val > upper_donch) and volume_confirmed
        # Short logic: price below weekly PP AND breaks below Donchian lower with volume confirmation
        short_condition = (close_val < pp_val) and (low_val < lower_donch) and volume_confirmed
        
        # Exit logic: 
        # Long exit: price crosses below weekly PP (trend change) OR retouches Donchian lower (failed breakout)
        long_exit = (position == 1 and (close_val < pp_val or low_val <= lower_donch))
        # Short exit: price crosses above weekly PP (trend change) OR retouches Donchian upper (failed breakout)
        short_exit = (position == -1 and (close_val > pp_val or high_val >= upper_donch))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0