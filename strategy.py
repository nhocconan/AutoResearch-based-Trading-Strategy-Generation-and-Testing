#!/usr/bin/env python3
name = "4h_HTF_1d_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Volume confirmation (4h volume > 1.5x 20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume
            if close[i] > upper_20_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume
            elif close[i] < lower_20_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Break below lower Donchian
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Break above upper Donchian
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals