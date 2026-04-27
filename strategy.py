#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-19:i+1])
        lower_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-day average volume
    avg_vol_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        avg_vol_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align Donchian channels and average volume to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian and volume data
    start_idx = 19
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(avg_vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        avg_vol = avg_vol_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with above-average volume
            if price > upper and vol > avg_vol:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian with above-average volume
            elif price < lower and vol > avg_vol:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to or below lower Donchian
            if price <= lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to or above upper Donchian
            if price >= upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0