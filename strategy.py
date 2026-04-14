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
    
    # Load daily data for price channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day Donchian channel
    donchian_high_20 = np.full_like(close_1d, np.nan)
    donchian_low_20 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            donchian_high_20[i] = np.max(high_1d[i-19:i+1])
            donchian_low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Align to 1d timeframe (same as input)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 20-week EMA for trend
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        alpha = 2.0 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w, additional_delay_bars=1)
    
    # Volume filter: 20-day average volume
    vol_ma_20 = np.full_like(close_1d, np.nan)
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-day average
        if vol_ma_20_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 20-day Donchian high + price > 20-week EMA + volume surge
            if (close[i] > donchian_high_20_aligned[i] and
                close[i] > ema_20_1w_aligned[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 20-day Donchian low + price < 20-week EMA + volume surge
            elif (close[i] < donchian_low_20_aligned[i] and
                  close[i] < ema_20_1w_aligned[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below 20-day Donchian low OR price < 20-week EMA
            if (close[i] < donchian_low_20_aligned[i] or 
                close[i] < ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above 20-day Donchian high OR price > 20-week EMA
            if (close[i] > donchian_high_20_aligned[i] or 
                close[i] > ema_20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian20_EMA20_Volume"
timeframe = "1d"
leverage = 1.0