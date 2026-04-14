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
    
    # Load 4h data for trend and price channel (primary trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4h Donchian channel (20-period)
    donchian_high_20 = np.full_like(close_4h, np.nan)
    donchian_low_20 = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= 20:
        for i in range(19, len(close_4h)):
            donchian_high_20[i] = np.max(high_4h[i-19:i+1])
            donchian_low_20[i] = np.min(low_4h[i-19:i+1])
    
    # 4h EMA (34-period) for trend filter
    ema_34_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 34:
        ema_34_4h[33] = np.mean(close_4h[:34])
        for i in range(34, len(close_4h)):
            ema_34_4h[i] = (close_4h[i] * 2 + ema_34_4h[i-1] * 32) / 34
    
    # Load 1d data for volume confirmation (HTF volume filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 1d volume MA (20-period)
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all 4h indicators to 4h timeframe (same as primary)
    donchian_high_20_4h = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_4h = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    ema_34_4h_4h = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_ma_20_1d_4h = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_4h[i]) or 
            np.isnan(donchian_low_20_4h[i]) or 
            np.isnan(ema_34_4h_4h[i]) or 
            np.isnan(vol_ma_20_1d_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 1d average volume
        if vol_ma_20_1d_4h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume_4h[i // 16] / vol_ma_20_1d_4h[i] if i >= 16 else 0
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high + price > 4h EMA34 + volume surge
            if (close[i] > donchian_high_20_4h[i] and
                close[i] > ema_34_4h_4h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 4h Donchian low + price < 4h EMA34 + volume surge
            elif (close[i] < donchian_low_20_4h[i] and
                  close[i] < ema_34_4h_4h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below 4h Donchian low OR price < 4h EMA34
            if (close[i] < donchian_low_20_4h[i] or 
                close[i] < ema_34_4h_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above 4h Donchian high OR price > 4h EMA34
            if (close[i] > donchian_high_20_4h[i] or 
                close[i] > ema_34_4h_4h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian20_EMA34_1dVolume"
timeframe = "4h"
leverage = 1.0