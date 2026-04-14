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
    
    # Load 12h data for price channel and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian channel (20-period)
    donchian_high_20 = np.full_like(close_12h, np.nan)
    donchian_low_20 = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= 20:
        for i in range(19, len(close_12h)):
            donchian_high_20[i] = np.max(high_12h[i-19:i+1])
            donchian_low_20[i] = np.min(low_12h[i-19:i+1])
    
    # 12h EMA (50-period) for trend
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 48) / 50
    
    # Load 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 1d volume MA (20-period)
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align to 12h timeframe
    donchian_high_20_12h = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_12h = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    ema_50_12h_12h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_20_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_12h[i]) or 
            np.isnan(donchian_low_20_12h[i]) or 
            np.isnan(ema_50_12h_12h[i]) or 
            np.isnan(vol_ma_20_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h volume vs 20-period 1d average volume
        if vol_ma_20_1d_12h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_12h[i] / vol_ma_20_1d_12h[i]
        
        if position == 0:
            # Long: Price breaks above 12h Donchian high + price > 12h EMA50 + volume surge
            if (close_12h[i] > donchian_high_20_12h[i] and
                close_12h[i] > ema_50_12h_12h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 12h Donchian low + price < 12h EMA50 + volume surge
            elif (close_12h[i] < donchian_low_20_12h[i] and
                  close_12h[i] < ema_50_12h_12h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below 12h Donchian low OR price < 12h EMA50
            if (close_12h[i] < donchian_low_20_12h[i] or 
                close_12h[i] < ema_50_12h_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above 12h Donchian high OR price > 12h EMA50
            if (close_12h[i] > donchian_high_20_12h[i] or 
                close_12h[i] > ema_50_12h_12h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian20_EMA50_Volume"
timeframe = "12h"
leverage = 1.0