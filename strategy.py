#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donchian_upper_4h = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_lower_4h = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need Donchian channels and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_4h[i]) or 
            np.isnan(donchian_lower_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume
            if (close[i] > donchian_upper_4h[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume
            elif (close[i] < donchian_lower_4h[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower
            if close[i] < donchian_lower_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper
            if close[i] > donchian_upper_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume"
timeframe = "4h"
leverage = 1.0