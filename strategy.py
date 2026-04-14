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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 10-period weekly EMA for trend
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(df_1w), np.nan)
    if len(close_1w) >= 10:
        alpha = 2.0 / (10 + 1)
        ema_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load daily data for price channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.full(len(df_1d), np.nan)
    donchian_low = np.full(len(df_1d), np.nan)
    
    if len(high_1d) >= 20:
        for i in range(19, len(df_1d)):
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    dh_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike detection (20-day average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(dh_aligned[i]) or 
            np.isnan(dl_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: require significant spike
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if volume_ratio < 1.5:  # Require at least 1.5x average volume
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high with weekly uptrend
            if close[i] > dh_aligned[i] and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 20-day low with weekly downtrend
            elif close[i] < dl_aligned[i] and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 20-day low or weekly trend turns down
            if close[i] < dl_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 20-day high or weekly trend turns up
            if close[i] > dh_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian20_WeeklyEMA10_Volume"
timeframe = "1d"
leverage = 1.0