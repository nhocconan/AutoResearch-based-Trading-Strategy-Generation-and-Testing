#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-week high/low)
    high_20w = np.full(len(df_1w), np.nan)
    low_20w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i >= 19:
            high_20w[i] = np.max(df_1w['high'].iloc[i-19:i+1])
            low_20w[i] = np.min(df_1w['low'].iloc[i-19:i+1])
    
    # Align weekly Donchian to 6h timeframe
    high_20w_6h = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_6h = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume confirmation: 4-period average (24h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(high_20w_6h[i]) or 
            np.isnan(low_20w_6h[i]) or 
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low
            if close[i] < low_20w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high
            if close[i] > high_20w_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above weekly Donchian high with volume confirmation
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > high_20w_6h[i] and 
                vol_ratio > 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below weekly Donchian low with volume confirmation
            elif (close[i] < low_20w_6h[i] and 
                  vol_ratio > 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals