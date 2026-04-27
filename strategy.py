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
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels
    donchian_high_1d = np.full(len(high_1d), np.nan)
    donchian_low_1d = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            donchian_high_1d[i] = np.max(high_1d[i-19:i+1])
            donchian_low_1d[i] = np.min(low_1d[i-19:i+1])
        else:
            donchian_high_1d[i] = np.nan
            donchian_low_1d[i] = np.nan
    
    # Calculate 20-day average volume
    avg_volume_1d = np.full(len(volume_1d), np.nan)
    for i in range(len(volume_1d)):
        if i >= 19:
            avg_volume_1d[i] = np.mean(volume_1d[i-19:i+1])
        else:
            avg_volume_1d[i] = np.nan
    
    # Align Donchian and volume to 6h timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 6-hour EMA50 for trend filter
    ema_period = 50
    ema_6h = np.full(n, np.nan)
    if n >= ema_period:
        ema_6h[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema_6h[i] = (close[i] * (2 / (ema_period + 1)) + 
                         ema_6h[i-1] * (1 - (2 / (ema_period + 1))))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume, and EMA
    start_idx = max(19, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(ema_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        donch_high = donchian_high_1d_aligned[i]
        donch_low = donchian_low_1d_aligned[i]
        ema_trend = ema_6h[i]
        
        if position == 0:
            # Long: Break above Donchian high with volume confirmation in uptrend
            if (price > donch_high and vol > 1.5 * avg_vol and price > ema_trend):
                signals[i] = size
                position = 1
            # Short: Break below Donchian low with volume confirmation in downtrend
            elif (price < donch_low and vol > 1.5 * avg_vol and price < ema_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to Donchian low or trend fails
            if price < donch_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to Donchian high or trend fails
            if price > donch_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6H_Donchian_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0