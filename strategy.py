#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_20 = np.full(len(high_1d), np.nan)
    lower_20 = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 50-period EMA on daily
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = close_1d[i] * (2 / (50 + 1)) + ema_50[i-1] * (1 - (2 / (50 + 1)))
    
    # Calculate volume ratio (current vs 20-day average)
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_ratio = volume_1d / vol_ma_20
    
    # Align indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA, and volume ratio
    start_idx = max(19, 49)  # 49 for EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio_val > 1.2
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume + above EMA
            if price > upper and vol_filter and price > ema_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian + volume + below EMA
            elif price < lower and vol_filter and price < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian or trend fails
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper Donchian or trend fails
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Donchian20_1dEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0