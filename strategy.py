#!/usr/bin/env python3
"""
4h_EquityCurveMomentum_VolumeFilter_TightEntry
Hypothesis: Uses equity curve momentum (price > 20-period high-low midpoint) with volume confirmation (>2x average) to capture sustained moves. Tight entry conditions limit trades to 20-30 per year, reducing fee drag while maintaining edge in both bull and bear markets via momentum persistence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high-low midpoint (equity curve proxy)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint = (highest_high + lowest_low) / 2
    
    # Volume confirmation: >2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(midpoint[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Momentum condition: price above/below midpoint
        above_mid = close[i] > midpoint[i]
        below_mid = close[i] < midpoint[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry conditions
        long_entry = above_mid and vol_confirm
        short_entry = below_mid and vol_confirm
        
        # Exit conditions: reverse momentum
        long_exit = below_mid
        short_exit = above_mid
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_EquityCurveMomentum_VolumeFilter_TightEntry"
timeframe = "4h"
leverage = 1.0