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
    
    # Get 1d data for Donchian and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels
    upper_channel = np.full(len(high_1d), np.nan)
    lower_channel = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_channel[i] = np.max(high_1d[i-20:i])
        lower_channel[i] = np.min(low_1d[i-20:i])
    
    # Calculate 20-period average volume
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        avg_volume_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = np.full(len(high_w), np.nan)
    r1 = np.full(len(high_w), np.nan)
    s1 = np.full(len(high_w), np.nan)
    r2 = np.full(len(high_w), np.nan)
    s2 = np.full(len(high_w), np.nan)
    r3 = np.full(len(high_w), np.nan)
    s3 = np.full(len(high_w), np.nan)
    
    for i in range(len(high_w)):
        pp = (high_w[i] + low_w[i] + close_w[i]) / 3.0
        pivot[i] = pp
        r1[i] = 2 * pp - low_w[i]
        s1[i] = 2 * pp - high_w[i]
        r2[i] = pp + (high_w[i] - low_w[i])
        s2[i] = pp - (high_w[i] - low_w[i])
        r3[i] = high_w[i] + 2 * (pp - low_w[i])
        s3[i] = low_w[i] - 2 * (high_w[i] - pp)
    
    # Align indicators to 6h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, volume average, and pivot points
    start_idx = max(20, 20) + 5  # buffer for calculations
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / avg_volume_1d_aligned[i] if avg_volume_1d_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above upper Donchian channel + volume spike + above weekly pivot
            if (price > upper_channel_aligned[i] and 
                vol_ratio > 1.5 and 
                price > pivot_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian channel + volume spike + below weekly pivot
            elif (price < lower_channel_aligned[i] and 
                  vol_ratio > 1.5 and 
                  price < pivot_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower Donchian channel OR RSI-like mean reversion
            if (price < lower_channel_aligned[i] or 
                price < pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper Donchian channel OR RSI-like mean reversion
            if (price > upper_channel_aligned[i] or 
                price > pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0