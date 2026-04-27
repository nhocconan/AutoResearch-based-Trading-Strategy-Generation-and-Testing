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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    high_20 = np.full(len(high_1d), np.nan)
    low_20 = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        high_20[i] = np.max(high_1d[i-19:i+1])
        low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 50-week EMA
    ema_50w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50w[i] = close_1w[i] * alpha + ema_50w[i-1] * (1 - alpha)
    
    # Get volume data for confirmation
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d_vol['volume'].values
    # Calculate 20-day average volume
    avg_vol_20 = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        avg_vol_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_50w_aligned = align_htf_to_ltf(prices, df_1w, ema_50w)
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d_vol, avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators
    start_idx = max(99, 49)  # 20-day Donchian needs 19, 50-week EMA needs 49
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50w_aligned[i]) or np.isnan(avg_vol_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_vol_20_aligned[i]
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        trend = ema_50w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation and uptrend
            if price > upper and vol_confirm and price > trend:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian band with volume confirmation and downtrend
            elif price < lower and vol_confirm and price < trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of channel or trend fails
            mid = (upper + lower) / 2
            if price < mid or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle of channel or trend fails
            mid = (upper + lower) / 2
            if price > mid or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4H_Donchian_Breakout_1W_EMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0