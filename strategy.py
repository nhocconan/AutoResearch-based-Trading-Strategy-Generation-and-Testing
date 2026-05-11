#!/usr/bin/env python3
name = "4h_Williams_Alligator_Trend_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def sma(arr, window):
    """Simple moving average with proper handling of NaN"""
    return pd.Series(arr).rolling(window=window, min_periods=window).mean().values

def williams_alligator(high, low, close):
    """Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)"""
    median_price = (high + low) / 2
    jaw = sma(median_price, 13)  # Blue line
    jaw = np.roll(jaw, 8)        # Shifted forward 8 bars
    
    teeth = sma(median_price, 8) # Red line
    teeth = np.roll(teeth, 5)    # Shifted forward 5 bars
    
    lips = sma(median_price, 5)  # Green line
    lips = np.roll(lips, 3)      # Shifted forward 3 bars
    
    return jaw, teeth, lips

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator components
    jaw, teeth, lips = williams_alligator(high_1d, low_1d, close_1d)
    
    # Align to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume filter: current volume > 1.5x daily average volume (aligned)
    vol_ma_1d = sma(volume_1d, 20)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_filter = volume > (vol_ma_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Wait for Alligator to form
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + volume
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + volume
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment breaks (lips <= teeth) or volume drops
            if (lips_aligned[i] <= teeth_aligned[i] or not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment breaks (lips >= teeth) or volume drops
            if (lips_aligned[i] >= teeth_aligned[i] or not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals