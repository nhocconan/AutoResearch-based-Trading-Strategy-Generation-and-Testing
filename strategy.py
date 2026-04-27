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
    
    # Get 12h data for structure
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20) channels
    donchian_high_12h = np.full(len(df_12h), np.nan)
    donchian_low_12h = np.full(len(df_12h), np.nan)
    
    for i in range(19, len(df_12h)):
        donchian_high_12h[i] = np.max(high_12h[i-19:i+1])
        donchian_low_12h[i] = np.min(low_12h[i-19:i+1])
    
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day volume SMA(20)
    vol_sma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_sma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Calculate 6h volume ratio (current / SMA20)
    vol_ratio_6h = np.full(n, np.nan)
    vol_sma_20_6h = np.full(n, np.nan)
    for i in range(19, n):
        vol_sma_20_6h[i] = np.mean(volume[i-19:i+1])
        if vol_sma_20_6h[i] > 0:
            vol_ratio_6h[i] = volume[i] / vol_sma_20_6h[i]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(19, 19) + 1  # 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_12h_aligned[i]) or 
            np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_break = price > donchian_high_12h_aligned[i]
        lower_break = price < donchian_low_12h_aligned[i]
        volume_confirm = vol_ratio_6h[i] > 1.5 and vol_sma_20_1d_aligned[i] > 0
        
        if position == 0:
            if upper_break and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif lower_break and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if price < donchian_low_12h_aligned[i]:  # Reverse signal
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.25
        elif position == -1:
            if price > donchian_high_12h_aligned[i]:  # Reverse signal
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_12hChannel_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0