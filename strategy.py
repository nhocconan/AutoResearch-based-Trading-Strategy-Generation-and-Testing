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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for trend) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 4h Donchian Channels (20-period) ===
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 1d Donchian Channels (20-period) for trend filter ===
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # === 4h volume ratio for confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian channels
    warmup = 25
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_high_1d_aligned[i]) or 
            np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        dh_1d = donchian_high_1d_aligned[i]
        dl_1d = donchian_low_1d_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price breaks below 4h Donchian low or 1d trend turns bearish
            if price < dl or close_4h[i] < dl_1d:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price breaks above 4h Donchian high or 1d trend turns bullish
            if price > dh or close_4h[i] > dh_1d:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above 4h Donchian high, above 1d Donchian low (bullish trend), volume confirmation
            if price > dh and close_4h[i] > dl_1d and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 4h Donchian low, below 1d Donchian high (bearish trend), volume confirmation
            elif price < dl and close_4h[i] < dh_1d and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Trend_Volume"
timeframe = "4h"
leverage = 1.0