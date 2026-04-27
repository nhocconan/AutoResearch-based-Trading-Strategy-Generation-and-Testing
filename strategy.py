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
    
    # Get 12h data for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    
    for i in range(19, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-19:i+1])
        donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian levels to main timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align volume MA to main timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 12h EMA20 for trend filter
    ema_period = 20
    ema_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(df_12h['close'].values[:ema_period])
        for i in range(ema_period, len(df_12h)):
            ema_12h[i] = (df_12h['close'].values[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align EMA to main timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume MA, and EMA
    start_idx = max(19, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema_trend = ema_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation in uptrend
            if price > upper and vol > vol_ma and price > ema_trend:
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low with volume confirmation in downtrend
            elif price < lower and vol > vol_ma and price < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price returns to Donchian low or trend fails
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price returns to Donchian high or trend fails
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0