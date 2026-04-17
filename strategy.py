#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_Volume
Strategy: 12-hour breakout of daily Camarilla R1/S1 with volume confirmation.
Long: Price breaks above daily R1 + volume > 1.8x 20-period avg
Short: Price breaks below daily S1 + volume > 1.8x 20-period avg
Exit: Opposite breakout (reverse signal)
Position size: 0.25
Designed to capture institutional breakout levels with volume confirmation.
Timeframe: 12h
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
    
    # Calculate daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 (daily)
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period average
        volume_filter = volume[i] > (1.8 * volume_ma20[i])
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_aligned[i-1]  # break above previous day R1
        breakout_s1 = close[i] < s1_aligned[i-1]  # break below previous day S1
        
        if position == 0:
            # Long: breakout above R1 + volume filter
            if breakout_r1 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter
            elif breakout_s1 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakout below S1 (reverse signal)
            if breakout_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above R1 (reverse signal)
            if breakout_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0