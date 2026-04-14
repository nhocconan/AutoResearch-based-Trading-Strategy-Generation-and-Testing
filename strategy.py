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
    
    # Load 1h data for price channel and volume
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # 1h Donchian channel (20-period)
    donchian_high_20 = np.full_like(close_1h, np.nan)
    donchian_low_20 = np.full_like(close_1h, np.nan)
    
    if len(close_1h) >= 20:
        for i in range(19, len(close_1h)):
            donchian_high_20[i] = np.max(high_1h[i-19:i+1])
            donchian_low_20[i] = np.min(low_1h[i-19:i+1])
    
    # 1h volume MA (20-period)
    vol_ma_20 = np.full_like(volume_1h, np.nan)
    if len(volume_1h) >= 20:
        for i in range(19, len(volume_1h)):
            vol_ma_20[i] = np.mean(volume_1h[i-19:i+1])
    
    # Align to 15m timeframe
    donchian_high_20_15m = align_htf_to_ltf(prices, df_1h, donchian_high_20)
    donchian_low_20_15m = align_htf_to_ltf(prices, df_1h, donchian_low_20)
    vol_ma_20_15m = align_htf_to_ltf(prices, df_1h, vol_ma_20)
    
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
    
    ema_50_4h_15m = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_15m[i]) or 
            np.isnan(donchian_low_20_15m[i]) or 
            np.isnan(vol_ma_20_15m[i]) or 
            np.isnan(ema_50_4h_15m[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 15m volume vs 20-period 1h average volume
        if vol_ma_20_15m[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_15m[i]
        
        if position == 0:
            # Long: Price breaks above 1h Donchian high + price > 4h EMA50 + volume surge
            if (close[i] > donchian_high_20_15m[i] and
                close[i] > ema_50_4h_15m[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 1h Donchian low + price < 4h EMA50 + volume surge
            elif (close[i] < donchian_low_20_15m[i] and
                  close[i] < ema_50_4h_15m[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below 1h Donchian low OR price < 4h EMA50
            if (close[i] < donchian_low_20_15m[i] or 
                close[i] < ema_50_4h_15m[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above 1h Donchian high OR price > 4h EMA50
            if (close[i] > donchian_high_20_15m[i] or 
                close[i] > ema_50_4h_15m[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_Donchian20_EMA50_Volume"
timeframe = "15m"
leverage = 1.0