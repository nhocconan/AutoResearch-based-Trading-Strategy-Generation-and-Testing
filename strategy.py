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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper[i] = np.max(high_1d[i-19:i+1])
        lower[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200
    ema_period = 200
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_period + 1)) + 
                        ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align EMA200 to 4h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA, and volume MA
    start_idx = max(20, 200, vol_period) + 25
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper + volume spike + price > 4h EMA200
            if (price > upper_1d_aligned[i] and 
                vol_ratio > 1.5 and 
                price > ema_4h_aligned[i]):
                signals[i] = size
                position = 1
            # Short: price breaks below 1d Donchian lower + volume spike + price < 4h EMA200
            elif (price < lower_1d_aligned[i] and 
                  vol_ratio > 1.5 and 
                  price < ema_4h_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 1d Donchian lower OR trend filter fails
            if (price < lower_1d_aligned[i] or 
                price < ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above 1d Donchian upper OR trend filter fails
            if (price > upper_1d_aligned[i] or 
                price > ema_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1d_Volume_EMA200"
timeframe = "4h"
leverage = 1.0