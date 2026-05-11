#!/usr/bin/env python3
name = "1d_1w_Donchian_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over past 20 weeks
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 weeks
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Volume filter: current volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian upper band + volume filter
            if close[i] > upper_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian lower band + volume filter
            elif close[i] < lower_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly Donchian lower band
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly Donchian upper band
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals