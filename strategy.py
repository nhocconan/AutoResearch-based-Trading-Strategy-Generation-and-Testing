#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: Camarilla pivot levels derived from 1d candles provide strong daily support/resistance.
Breakouts above R4 or below S4 with volume confirmation indicate institutional participation and trend continuation.
This strategy works in both bull and bear markets by capturing breakouts in the direction of the trend.
Target: 12-37 trades/year per symbol.
"""

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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]  # first bar uses its own close
    
    range_1d = high_1d - low_1d
    
    # Resistance levels
    R1 = close_prev + (range_1d * 1.0833 / 12)
    R2 = close_prev + (range_1d * 1.1666 / 6)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    R4 = close_prev + (range_1d * 1.5000 / 2)
    
    # Support levels
    S1 = close_prev - (range_1d * 1.0833 / 12)
    S2 = close_prev - (range_1d * 1.1666 / 6)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    S4 = close_prev - (range_1d * 1.5000 / 2)
    
    # Align levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R4 with volume expansion
        long_breakout = close[i] > R4_aligned[i] and volume_expansion[i]
        
        # Short breakdown: price breaks below S4 with volume expansion
        short_breakout = close[i] < S4_aligned[i] and volume_expansion[i]
        
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

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0