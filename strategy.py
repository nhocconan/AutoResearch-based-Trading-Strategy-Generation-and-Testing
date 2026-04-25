#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d Williams %R filter + volume confirmation
Hypothesis: Donchian breakouts capture strong momentum. Williams %R on daily chart filters for overextended conditions - 
we only take breakouts when daily Williams %R is not in extreme overbought/oversold territory (> -20 for longs, < -80 for shorts).
This avoids buying tops and selling bottoms. Volume confirmation ensures breakout validity. Works in both bull and bear markets
by capturing momentum shifts while avoiding counter-trend entries at extremes.
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
    
    # Get 1d data for Williams %R (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on daily chart
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    williams_r = np.full(len(daily_close), np.nan)
    for i in range(13, len(daily_close)):
        highest_high = np.max(daily_high[i-13:i+1])
        lowest_low = np.min(daily_low[i-13:i+1])
        if highest_high != lowest_low:
            williams_r[i] = ((highest_high - daily_close[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate Donchian channels (20-period) on 6h chart
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, Williams %R, and volume MA to propagate
    start_idx = max(19, 13, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        williams_r_val = williams_r_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        volume_spike = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high AND Williams %R not overbought (> -20) AND volume spike
            long_condition = (curr_close > donch_high) and (williams_r_val > -20) and volume_spike
            # Short: price breaks below Donchian low AND Williams %R not oversold (< -80) AND volume spike
            short_condition = (curr_close < donch_low) and (williams_r_val < -80) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price breaks below Donchian low (reversal signal)
            if curr_close < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high (reversal signal)
            if curr_close > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWilliamsR_Filter_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0