#!/usr/bin/env python3
"""
4h_Donchian_20_With_Volume_Confirmation
Hypothesis: Trade 4h Donchian channel breakout with volume confirmation.
Long when price breaks above 20-period high + volume > 1.5x avg volume.
Short when price breaks below 20-period low + volume > 1.5x avg volume.
Exit on opposite breakout or volume drop.
Donchian channels provide clear trend structure, volume filters false breakouts.
Target: 80-150 total trades over 4 years (20-37/year) with position size 0.25.
Works in bull/bear: volume surge confirms institutional interest in breakouts.
"""

name = "4h_Donchian_20_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(volume[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high + volume confirmation
            if close[i] > highest_high[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume confirmation
            elif close[i] < lowest_low[i] and volume[i] > 1.5 * avg_volume[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low OR volume drops below average
            if close[i] < lowest_low[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR volume drops below average
            if close[i] > highest_high[i] or volume[i] < avg_volume[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals