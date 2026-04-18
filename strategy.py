#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume
Camarilla pivot breakout strategy with volume confirmation:
- Long when price breaks above R1 level with volume > 1.5x 20-period average
- Short when price breaks below S1 level with volume > 1.5x 20-period average
- Exit when price crosses back through the central pivot point (PP)
- Uses 1d data for Camarilla levels (R1, S1, PP)
- Designed for 15-25 trades/year per symbol
Works in both bull (captures R1 breakouts) and bear (captures S1 breakdowns) markets
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1, PP) from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = (high_1d + low_1d + close_1d) / 3
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 level + volume confirmation
            if close[i] > r1_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 level + volume confirmation
            elif close[i] < s1_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point (PP)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0  # exit long
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point (PP)
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0  # exit short
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0