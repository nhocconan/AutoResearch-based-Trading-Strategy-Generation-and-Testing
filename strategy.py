#!/usr/bin/env python3
"""
1d_Weekly_Camarilla_Breakout_VolumeFilter
Strategy: Daily price breakout at weekly Camarilla pivot levels with volume confirmation.
Long: Price breaks above weekly Camarilla R1 + volume > 1.8x daily average
Short: Price breaks below weekly Camarilla S1 + volume > 1.8x daily average
Exit: Price returns to weekly Camarilla pivot midpoint
Position size: 0.25
Designed to capture institutional breakout levels with volume validation.
Timeframe: 1d
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
    
    # Calculate weekly Camarilla pivot levels from weekly OHLC
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    r1 = close_1w + (range_1w * 1.1 / 12)
    s1 = close_1w - (range_1w * 1.1 / 12)
    midpoint = (r1 + s1) / 2
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    midpoint_aligned = align_htf_to_ltf(prices, df_1w, midpoint)
    
    # Volume confirmation (20-day average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(midpoint_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-day average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i-1]  # break above previous day's R1
        breakout_down = close[i] < s1_aligned[i-1]  # break below previous day's S1
        
        # Return to midpoint
        return_to_mid = abs(close[i] - midpoint_aligned[i]) < 0.15 * (r1_aligned[i] - s1_aligned[i])
        
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
            # Exit long: return to midpoint or break below S1
            if return_to_mid or breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to midpoint or break above R1
            if return_to_mid or breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Camarilla_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0