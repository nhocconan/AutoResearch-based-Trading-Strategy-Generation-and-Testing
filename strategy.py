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
    
    # Load daily data for structure (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(50) for trend filter
    ema50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha + ema50_1d[i-1] * (1 - alpha)
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily Donchian(20) for breakout levels
    donch_high_20 = np.full_like(close_1d, np.nan)
    donch_low_20 = np.full_like(close_1d, np.nan)
    if len(high_1d) >= 20 and len(low_1d) >= 20:
        for i in range(19, len(high_1d)):
            donch_high_20[i] = np.max(high_1d[i-19:i+1])
            donch_low_20[i] = np.min(low_1d[i-19:i+1])
    
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate 6-hour volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + above EMA50 + volume surge
            if (close[i] > donch_high_20_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                volume_ratio > 1.8):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian low + below EMA50 + volume surge
            elif (close[i] < donch_low_20_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  volume_ratio > 1.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price breaks below Donchian low OR below EMA50
            if (close[i] < donch_low_20_aligned[i] or 
                close[i] < ema50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price breaks above Donchian high OR above EMA50
            if (close[i] > donch_high_20_aligned[i] or 
                close[i] > ema50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Donchian20_EMA50_Volume_Breakout"
timeframe = "6h"
leverage = 1.0