#!/usr/bin/env python3
# 2025-06-22 | 4h_Donchian20_Breakout_1dEMA200_Trend_VolumeSpike
# Hypothesis: Breakouts from 4h Donchian(20) channels with 1d EMA200 trend filter and volume spike confirmation.
# The Donchian channel provides clear support/resistance levels. EMA200 filters for long-term trend direction.
# Volume spike (>2x 24-period average) confirms breakout strength. Designed for low trade frequency.
# EMA200 is effective in both bull and bear markets as a dynamic support/resistance level.

name = "4h_Donchian20_Breakout_1dEMA200_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (ema_200_1d[i-1] * 199 + close_1d[i]) / 200
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 4h Donchian channel (20-period)
    high_max = np.full_like(high, np.nan)
    low_min = np.full_like(low, np.nan)
    if len(high) >= 20:
        for i in range(19, len(high)):
            high_max[i] = np.max(high[i-19:i+1])
            low_min[i] = np.min(low[i-19:i+1])
    
    # Volume spike filter: current volume / 24-period average volume (24*4h = 4 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 200)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND uptrend (price > EMA200) AND volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below Donchian low AND downtrend (price < EMA200) AND volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend reversal (price < EMA200)
            if close[i] < low_min[i] or close[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend reversal (price > EMA200)
            if close[i] > high_max[i] or close[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals