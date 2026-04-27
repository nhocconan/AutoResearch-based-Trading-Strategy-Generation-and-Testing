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
    
    # Get weekly data for trend filter (EWMA of close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    alpha_w = 2 / (20 + 1)
    ewma_1w_20 = np.full(len(df_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 19:
            ewma_1w_20[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ewma_1w_20[i-1]):
                ewma_1w_20[i] = np.mean(close_1w[i-19:i+1])
            else:
                ewma_1w_20[i] = close_1w[i] * alpha_w + ewma_1w_20[i-1] * (1 - alpha_w)
    
    ewma_1w_20_aligned = align_htf_to_ltf(prices, df_1w, ewma_1w_20)
    
    # Get daily data for Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high_20 = np.full(len(df_1d), np.nan)
    donchian_low_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donchian_high_20[i] = np.max(high_1d[i-20:i])
        donchian_low_20[i] = np.min(low_1d[i-20:i])
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 20)  # daily Donchian needs 20, weekly EMA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ewma_1w_20_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume (strict to reduce trades)
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price breaks above 20-day Donchian high with volume and weekly uptrend
            if (volume_confirmation and 
                price > donchian_high_20_aligned[i] and 
                ewma_1w_20_aligned[i] > ewma_1w_20_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day Donchian low with volume and weekly downtrend
            elif (volume_confirmation and 
                  price < donchian_low_20_aligned[i] and 
                  ewma_1w_20_aligned[i] < ewma_1w_20_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 20-day Donchian low or weekly trend turns down
            if (price < donchian_low_20_aligned[i] or 
                ewma_1w_20_aligned[i] < ewma_1w_20_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price breaks above 20-day Donchian high or weekly trend turns up
            if (price > donchian_high_20_aligned[i] or 
                ewma_1w_20_aligned[i] > ewma_1w_20_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_Donchian20_WeeklyEWMA20_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0