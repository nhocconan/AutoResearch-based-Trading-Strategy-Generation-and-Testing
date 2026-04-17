#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter
Strategy: 6-hour breakout of weekly pivot R1/S1 levels with volume confirmation.
Long: Price breaks above weekly pivot R1 + volume > 1.8x 20-period average
Short: Price breaks below weekly pivot S1 + volume > 1.8x 20-period average
Exit: Price returns to weekly pivot point (PP)
Position size: 0.25
Weekly pivot levels provide institutional reference points that work in both trending and ranging markets.
Timeframe: 6h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points (PP, R1, S1)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculation: PP = (H + L + C)/3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = (2 * PP) - L
    r1_1w = (2 * pp_1w) - low_1w
    # S1 = (2 * PP) - H
    s1_1w = (2 * pp_1w) - high_1w
    
    # Align weekly levels to 6h timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation (20-period MA on 6h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > r1_1w_aligned[i-1]  # break above weekly R1
        breakout_down = close[i] < s1_1w_aligned[i-1]  # break below weekly S1
        
        # Return to weekly pivot point for exit
        return_to_pp = abs(close[i] - pp_1w_aligned[i]) < 0.005 * close[i]  # within 0.5% of PP
        
        if position == 0:
            # Long: breakout above R1 + volume filter
            if breakout_up and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter
            elif breakout_down and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to PP or break below S1
            if return_to_pp or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to PP or break above R1
            if return_to_pp or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0