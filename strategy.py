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
    
    # Get weekly data for trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i < 33:
            ema_1w_34[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha_w + ema_1w_34[i-1] * (1 - alpha_w)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Get daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels on daily
    donchian_high_20 = np.full(len(df_1d), np.nan)
    donchian_low_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donchian_high_20[i] = np.max(high_1d[i-19:i+1])
        donchian_low_20[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate average volume(20) on daily
    avg_volume_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        avg_volume_20[i] = np.mean(volume_1d[i-19:i+1])
    
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(avg_volume_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume filter: current volume > 1.5 * average daily volume
        volume_filter = volume[i] > 1.5 * avg_volume_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume
            if (price > donchian_high_20_aligned[i] and 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume
            elif (price < donchian_low_20_aligned[i] and 
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly trend turns down
            if (price < donchian_low_20_aligned[i] or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly trend turns up
            if (price > donchian_high_20_aligned[i] or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_WeeklyEMA34_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0