#!/usr/bin/env python3
"""
6h_12h_Range_Breakout_With_Volume_Confirmation
Hypothesis: The 12-hour range provides stronger support/resistance than daily due to fewer false breaks.
Breakouts above the 12h high or below the 12h low with volume expansion capture momentum.
Works in both bull and bear markets by trading breakouts regardless of direction.
Target: 15-25 trades/year per symbol.
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
    
    # Get 12h data for range calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h high and low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Align 12h high/low to 6h timeframe
    high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_12h)
    low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(high_12h_aligned[i]) or np.isnan(low_12h_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above 12h high with volume expansion
        long_breakout = close[i] > high_12h_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below 12h low with volume expansion
        short_breakout = close[i] < low_12h_aligned[i] and volume_expansion[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_12h_Range_Breakout_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0