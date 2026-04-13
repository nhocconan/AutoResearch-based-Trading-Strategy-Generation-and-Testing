#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Volume
Hypothesis: Camarilla pivot levels derived from 12h candles provide strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation indicate institutional participation.
Works in both bull and bear markets by capturing breakouts in the direction of momentum.
Target: 20-30 trades/year per symbol.
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
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h < 2):
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formulas
    close_prev = np.roll(close_12h, 1)
    close_prev[0] = close_12h[0]
    
    range_12h = high_12h - low_12h
    
    # Resistance levels
    R3 = close_prev + (range_12h * 1.2500 / 4)
    R4 = close_prev + (range_12h * 1.5000 / 2)
    
    # Support levels
    S3 = close_prev - (range_12h * 1.2500 / 4)
    S4 = close_prev - (range_12h * 1.5000 / 2)
    
    # Align levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R3 with volume expansion
        long_breakout = close[i] > R3_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below S3 with volume expansion
        short_breakout = close[i] < S3_aligned[i] and volume_expansion[i]
        
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

name = "4h_12h_Camarilla_Pivot_Breakout_Volume"
timeframe = "4h"
leverage = 1.0