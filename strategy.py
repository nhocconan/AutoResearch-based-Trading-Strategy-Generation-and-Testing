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
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 20-period high and low for Donchian channel (weekly)
    donchian_high_20 = np.full_like(close_1w, np.nan)
    donchian_low_20 = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 20:
        for i in range(19, len(close_1w)):
            donchian_high_20[i] = np.max(high_1w[i-19:i+1])
            donchian_low_20[i] = np.min(low_1w[i-19:i+1])
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate 20-period volume average (weekly)
    vol_ma_20 = np.full_like(volume_1w, np.nan)
    if len(volume_1w) >= 20:
        for i in range(19, len(volume_1w)):
            vol_ma_20[i] = np.mean(volume_1w[i-19:i+1])
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 1d volume vs 20-period weekly average volume
        if vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + volume surge
            if (close[i] > donchian_high_20_aligned[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below weekly Donchian low + volume surge
            elif (close[i] < donchian_low_20_aligned[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below weekly Donchian low
            if close[i] < donchian_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above weekly Donchian high
            if close[i] > donchian_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian20_VolumeBreakout"
timeframe = "1d"
leverage = 1.0