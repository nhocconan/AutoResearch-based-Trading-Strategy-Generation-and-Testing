#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Daily Close for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Donchian channels (20-day period)
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20.values)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20.values)
    
    # Volume ratio: current vs 20-period median
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_ratio = volume / vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            continue
        
        # Long: price breaks above 1d Donchian high + volume spike
        if close[i] > highest_20_aligned[i] and vol_ratio[i] > 2.0:
            signals[i] = 0.30
        
        # Short: price breaks below 1d Donchian low + volume spike
        elif close[i] < lowest_20_aligned[i] and vol_ratio[i] > 2.0:
            signals[i] = -0.30
        
        # Exit: price returns to midline of Donchian channel
        elif i > 0 and signals[i-1] != 0:
            mid = (highest_20_aligned[i] + lowest_20_aligned[i]) / 2.0
            if (signals[i-1] > 0 and close[i] < mid) or (signals[i-1] < 0 and close[i] > mid):
                signals[i] = 0.0
        
        # Otherwise hold position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0