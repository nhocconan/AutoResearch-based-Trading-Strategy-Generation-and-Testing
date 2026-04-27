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
    # Calculate EMA(34) with proper initialization
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w_34[i] = close_1w[i]
        elif i < 34:
            ema_1w_34[i] = np.mean(close_1w[:i+1])
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha + ema_1w_34[i-1] * (1 - alpha)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Get daily data for price channel and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) channels
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 20-period average volume
    avg_volume_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        avg_volume_20[i] = np.mean(volume_1d[i-19:i+1])
    
    avg_volume_20_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_20)
    
    signals = np.zeros(n)
    position = 0
    
    # Start after all indicators are ready
    start_idx = max(34, 19)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1w_34_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume confirmation: current volume > 1.5 * average volume
        volume_spike = volume[i] > 1.5 * avg_volume_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + weekly uptrend
            if (price > donchian_high_aligned[i] and 
                volume_spike and 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + weekly downtrend
            elif (price < donchian_low_aligned[i] and 
                  volume_spike and 
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly trend turns down
            if (price < donchian_low_aligned[i] or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly trend turns up
            if (price > donchian_high_aligned[i] or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_VolumeSpike_WeeklyEMA34_v1"
timeframe = "1d"
leverage = 1.0