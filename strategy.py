#!/usr/bin/env python3
"""
6h 3-Day Range Breakout with Volume Confirmation
Long when price breaks above 3-day high with above-average volume
Short when price breaks below 3-day low with above-average volume
Exit when price returns to middle of 3-day range
Works in trending markets (both bull and bear) by capturing breakouts with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_3day_range_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 3-day lookback (12 bars for 6h timeframe: 3 days * 4 bars/day)
    lookback = 12
    
    # Rolling max/min for 3-day range
    high_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Middle of range for exit signal
    range_mid = (high_max + low_min) / 2.0
    
    # Volume confirmation - 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to middle of range
            if close[i] <= range_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle of range
            if close[i] >= range_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: breakout of 3-day range with volume confirmation
            if close[i] > high_max[i]:
                # Break above 3-day high -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < low_min[i]:
                # Break below 3-day low -> short
                position = -1
                signals[i] = -0.25
    
    return signals