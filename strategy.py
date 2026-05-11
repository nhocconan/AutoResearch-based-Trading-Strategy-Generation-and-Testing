#!/usr/bin/env python3
name = "6h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian(20) for trend filter (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h timeframe (use previous day's levels)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # 6h Donchian(20) breakout levels (current period)
    donchian_high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or \
           np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above 6h Donchian high + above 1d Donchian high + volume
            if close[i] > donchian_high_6h[i] and close[i] > donchian_high_20_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 6h Donchian low + below 1d Donchian low + volume
            elif close[i] < donchian_low_6h[i] and close[i] < donchian_low_20_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below 6h Donchian low
            if close[i] < donchian_low_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above 6h Donchian high
            if close[i] > donchian_high_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals