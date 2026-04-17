#!/usr/bin/env python3
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
    
    # Get daily data for Williams Alligator (1d)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13-period, offset 8), Teeth (8-period, offset 5), Lips (5-period, offset 3)
    jaw_len, teeth_len, lips_len = 13, 8, 5
    jaw_offset, teeth_offset, lips_offset = 8, 5, 3
    
    # Calculate smoothed median prices
    median_price_1d = (high_1d + low_1d) / 2
    
    # Jaw: SMMA of median price (13 periods, offset 8)
    jaw = pd.Series(median_price_1d).rolling(window=jaw_len, min_periods=jaw_len).mean()
    jaw = jaw.shift(jaw_offset)
    
    # Teeth: SMMA of median price (8 periods, offset 5)
    teeth = pd.Series(median_price_1d).rolling(window=teeth_len, min_periods=teeth_len).mean()
    teeth = teeth.shift(teeth_offset)
    
    # Lips: SMMA of median price (5 periods, offset 3)
    lips = pd.Series(median_price_1d).rolling(window=lips_len, min_periods=lips_len).mean()
    lips = lips.shift(lips_offset)
    
    # Align Alligator lines to 6h timeframe
    jaw_6h = align_htf_to_ltf(prices, df_1d, jaw.values)
    teeth_6h = align_htf_to_ltf(prices, df_1d, teeth.values)
    lips_6h = align_htf_to_ltf(prices, df_1d, lips.values)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(jaw_len + jaw_offset, teeth_len + teeth_offset, lips_len + lips_offset, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        if position == 0:
            # Long: Lips above Teeth above Teeth (bullish alignment) with volume
            if lips_6h[i] > teeth_6h[i] and teeth_6h[i] > jaw_6h[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth below Jaw (bearish alignment) with volume
            elif lips_6h[i] < teeth_6h[i] and teeth_6h[i] < jaw_6h[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips crosses below Teeth (bullish alignment broken)
            if lips_6h[i] < teeth_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips crosses above Teeth (bearish alignment broken)
            if lips_6h[i] > teeth_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_Alignment_VolumeFilter"
timeframe = "6h"
leverage = 1.0