#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Daily (1d) Camarilla pivot levels (R3/S3) act as key support/resistance on the 4h chart.
Breakouts above R3 or below S3 with volume expansion (>1.5x 20-period average) capture
institutional order flow. Works in both bull and bear markets by trading breakouts with volume
confirmation, reducing false signals. Target: 20-30 trades/year per symbol.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]
    
    # Daily range
    range_1d = high_1d - low_1d
    
    # Resistance levels (R3)
    R3 = close_prev + (range_1d * 1.2500 / 4)
    
    # Support levels (S3)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    
    # Align levels to 4h timeframe (already delayed by one bar via align_htf_to_ltf)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
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

name = "4h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0