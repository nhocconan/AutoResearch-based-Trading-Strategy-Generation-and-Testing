#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_SMVolume_Confirmation_v1
Hypothesis: Combines daily Camarilla pivot breakouts with smoothed volume confirmation (3-period SMA of volume ratio) to reduce whipsaws.
Uses breakout logic only when volume confirmation is above threshold, reducing false signals. Works in both bull and bear markets
by trading breakouts in either direction. Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
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
    
    # Previous close for pivot calculation
    close_prev = np.roll(close_1d, 1)
    close_prev[0] = close_1d[0]
    
    range_1d = high_1d - low_1d
    
    # Resistance (R3) and Support (S3) levels
    R3 = close_prev + (range_1d * 1.2500 / 4)
    S3 = close_prev - (range_1d * 1.2500 / 4)
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Smoothed volume confirmation: 3-period SMA of volume / 20-period volume average
    vol_series = pd.Series(volume)
    vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma_20
    vol_ratio_smooth = vol_ratio.rolling(window=3, min_periods=3).mean()
    volume_confirm = vol_ratio_smooth > 1.3  # Require sustained volume strength
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price crosses above R3 with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_confirm[i]
        
        # Short breakdown: price crosses below S3 with volume confirmation
        short_breakout = close[i] < S3_aligned[i] and volume_confirm[i]
        
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

name = "4h_1d_Camarilla_Pivot_Breakout_SMVolume_Confirmation_v1"
timeframe = "4h"
leverage = 1.0