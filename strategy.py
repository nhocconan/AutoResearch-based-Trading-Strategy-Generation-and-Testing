#!/usr/bin/env python3
"""
6h_WeeklyPivotDirection_DonchianBreakout_VolumeFilter
Hypothesis: Use weekly pivot point direction (from previous week's PP) as trend filter for 6h Donchian(20) breakouts with volume confirmation.
In bull markets: price above weekly PP, buy Donchian high breakouts with volume spike.
In bear markets: price below weekly PP, sell Donchian low breakouts with volume spike.
Weekly pivot provides institutional reference point, Donchian captures breakouts, volume confirms institutional participation.
Designed for low trade frequency (15-35/year) to minimize fee drag while capturing strong directional moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly high, low, close for pivot point calculation (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Point (PP) and support/resistance levels
    # PP = (H + L + C) / 3
    weekly_pp = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1 = 2*PP - L, S1 = 2*PP - H
    weekly_r1 = 2 * weekly_pp - low_1w
    weekly_s1 = 2 * weekly_pp - high_1w
    
    # Align weekly levels to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pp_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pp = weekly_pp_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > weekly PP AND price breaks above Donchian upper with volume spike
            if price > pp and price > upper_channel and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price < weekly PP AND price breaks below Donchian lower with volume spike
            elif price < pp and price < lower_channel and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower OR price crosses below weekly PP
            if price < lower_channel or price < pp:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR price crosses above weekly PP
            if price > upper_channel or price > pp:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivotDirection_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0