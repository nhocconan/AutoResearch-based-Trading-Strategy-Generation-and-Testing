#!/usr/bin/env python3
"""
1d_1w_Donchian_20_WeeklyTrend_4hVolume
Hypothesis: Uses daily Donchian(20) breakout with weekly trend filter and 4h volume confirmation to capture strong trends while avoiding whipsaws in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.zeros_like(close)
    donchian_low = np.zeros_like(close)
    
    for i in range(len(high_1d)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.zeros_like(close_1w)
    ema_20_1w[:] = np.nan
    if len(close_1w) >= 20:
        k = 2 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = close_1w[i] * k + ema_20_1w[i-1] * (1 - k)
    
    # Calculate 4h volume MA20 for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = np.zeros_like(volume_4h)
    vol_ma_20_4h[:] = np.nan
    if len(volume_4h) >= 20:
        for i in range(len(volume_4h)):
            if i < 19:
                vol_ma_20_4h[i] = np.mean(volume_4h[0:i+1]) if i >= 0 else volume_4h[i]
            else:
                vol_ma_20_4h[i] = np.mean(volume_4h[i-19:i+1])
    
    # Align indicators to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above Donchian high with weekly uptrend and volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume[i] > vol_ma_20_4h_aligned[i]):
                signals[i] = 0.30
                position = 1
                bars_since_entry = 0
            # Short: price breaks below Donchian low with weekly downtrend and volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume[i] > vol_ma_20_4h_aligned[i]):
                signals[i] = -0.30
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: price breaks below Donchian low or weekly trend turns down
            if close[i] < donchian_low_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price breaks above Donchian high or weekly trend turns up
            if close[i] > donchian_high_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_1w_Donchian_20_WeeklyTrend_4hVolume"
timeframe = "1d"
leverage = 1.0