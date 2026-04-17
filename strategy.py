#!/usr/bin/env python3
"""
4h_Williams_Alligator_Volume_Filter
Strategy: Williams Alligator (SMMA-based) on 4h with 1d volume confirmation.
Long: Green Alligator (jaw < teeth < lips) + price > lips + volume > 1.5x 20-day avg.
Short: Red Alligator (jaw > teeth > lips) + price < lips + volume > 1.5x 20-day avg.
Exit: Opposite Alligator alignment or volume filter fails.
Position size: 0.25
Designed to trend-follow with smoothed SMMA to reduce whipsaw, volume to confirm strength.
Timeframe: 4h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value: SMA
    result[length-1] = np.mean(source[:length])
    # Subsequent: SMMA = (prev_smma * (length-1) + current) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator parameters (13,8,5) with offsets (8,5,3)
    jaw_len, teeth_len, lips_len = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    # Calculate SMMA for median price (typical price)
    typical_price = (high + low + close) / 3.0
    jaw = smma(typical_price, jaw_len)
    teeth = smma(typical_price, teeth_len)
    lips = smma(typical_price, lips_len)
    
    # Apply offsets (shift right by offset bars)
    jaw = np.roll(jaw, jaw_offset)
    teeth = np.roll(teeth, teeth_offset)
    lips = np.roll(lips, lips_offset)
    # Set initial values to NaN after roll
    jaw[:jaw_offset] = np.nan
    teeth[:teeth_offset] = np.nan
    lips[:lips_offset] = np.nan
    
    # 1d volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(jaw_len + jaw_offset, teeth_len + teeth_offset, lips_len + lips_offset, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        volume_filter = vol_1d[i // 24] > (1.5 * vol_ma20_1d_aligned[i]) if i // 24 < len(vol_1d) else False
        
        # Alligator alignment
        green = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])  # jaw < teeth < lips
        red = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])   # jaw > teeth > lips
        
        # Entry conditions
        if position == 0:
            # Long: Green Alligator + price > lips + volume
            if green and (close[i] > lips[i]) and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Red Alligator + price < lips + volume
            elif red and (close[i] < lips[i]) and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Red Alligator or volume filter fails
            if red or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Green Alligator or volume filter fails
            if green or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Volume_Filter"
timeframe = "4h"
leverage = 1.0