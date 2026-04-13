#!/usr/bin/env python3
"""
6h_1w_Daily_Range_Breakout_With_Volume_Confirmation
Hypothesis: Weekly range (H-L) from prior week defines key support/resistance.
Breakouts above weekly high or below weekly low on 6h chart with volume expansion
capture institutional moves. Works in both bull and bear markets by trading
breakouts regardless of direction. Target: 15-25 trades/year per symbol.
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
    
    # Get weekly data for range calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly levels to 6h timeframe (previous week's levels)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above weekly high with volume expansion
        long_breakout = close[i] > weekly_high_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below weekly low with volume expansion
        short_breakout = close[i] < weekly_low_aligned[i] and volume_expansion[i]
        
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

name = "6h_1w_Daily_Range_Breakout_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0