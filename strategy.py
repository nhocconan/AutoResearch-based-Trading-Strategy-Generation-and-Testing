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
    
    # Load 1d data for price channel and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Donchian channel (20-period)
    donchian_high_20 = np.full_like(close_1d, np.nan)
    donchian_low_20 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            donchian_high_20[i] = np.max(high_1d[i-19:i+1])
            donchian_low_20[i] = np.min(low_1d[i-19:i+1])
    
    # 1d volume MA (20-period)
    vol_ma_20 = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align to 4h timeframe
    donchian_high_20_4h = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_4h = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    vol_ma_20_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Load 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h EMA (50-period) for trend
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (close_4h[i] * 2 + ema_50_4h[i-1] * 48) / 50
    
    ema_50_4h_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_4h[i]) or 
            np.isnan(donchian_low_20_4h[i]) or 
            np.isnan(vol_ma_20_4h[i]) or 
            np.isnan(ema_50_4h_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 20-period 1d average volume
        if vol_ma_20_4h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_4h[i]
        
        if position == 0:
            # Long: Price breaks above 1d Donchian high + price > 4h EMA50 + volume surge
            if (close[i] > donchian_high_20_4h[i] and
                close[i] > ema_50_4h_4h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 1d Donchian low + price < 4h EMA50 + volume surge
            elif (close[i] < donchian_low_20_4h[i] and
                  close[i] < ema_50_4h_4h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below 1d Donchian low OR price < 4h EMA50
            if (close[i] < donchian_low_20_4h[i] or 
                close[i] < ema_50_4h_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above 1d Donchian high OR price > 4h EMA50
            if (close[i] > donchian_high_20_4h[i] or 
                close[i] > ema_50_4h_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian20_EMA50_Volume"
timeframe = "4h"
leverage = 1.0