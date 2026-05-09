#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Breakouts from daily Donchian(20) channels with weekly trend filter and volume spike confirmation.
Uses 4h timeframe for execution, weekly EMA200 for trend direction, and volume > 2x 24-period average.
Designed for low trade frequency (20-50/year) to minimize fee drift and work in both bull and bear markets.
"""

name = "4h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (ema_200_1w[i-1] * 199 + close_1w[i]) / 200
    
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily Donchian channels (20-period high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels
    upper_20 = np.full_like(high_1d, np.nan)
    lower_20 = np.full_like(low_1d, np.nan)
    
    for i in range(19, len(high_1d)):
        upper_20[i] = np.max(high_1d[i-19:i+1])
        lower_20[i] = np.min(low_1d[i-19:i+1])
    
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
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
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above upper Donchian AND uptrend (price > EMA200) AND volume spike
            if (close[i] > upper_20_aligned[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below lower Donchian AND downtrend (price < EMA200) AND volume spike
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 4 bars
            if bars_since_entry < 4:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below lower Donchian OR trend reversal (price < EMA200)
                if close[i] < lower_20_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 4 bars
            if bars_since_entry < 4:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above upper Donchian OR trend reversal (price > EMA200)
                if close[i] > upper_20_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals