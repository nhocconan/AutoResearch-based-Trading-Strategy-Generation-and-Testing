#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Donchian(20) breakout + weekly pivot direction + volume confirmation.
# Long when price breaks above 1d Donchian upper band AND 1w pivot level above current price AND 6h volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian lower band AND 1w pivot level below current price AND 6h volume > 1.5x 20-period average.
# Exit when price crosses back inside the 1d Donchian channel.
# This captures breakouts aligned with weekly pivot levels, filtering false breakouts with volume confirmation.
# Designed to work in both bull and bear markets by using pivot levels for direction and volume for confirmation.
# Target: 12-37 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channel
    donchian_up = np.full_like(high_1d, np.nan)
    donchian_down = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        donchian_up[i] = np.max(high_1d[i-19:i+1])
        donchian_down[i] = np.min(low_1d[i-19:i+1])
    
    # Load 1w data for pivot point calculation (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pivot_1w = np.full_like(high_1w, np.nan)
    for i in range(len(high_1w)):
        pivot_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
    
    # Load 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_6h, np.nan)
    for i in range(19, len(volume_6h)):
        vol_ma_20[i] = np.mean(volume_6h[i-19:i+1])
    
    # Align indicators to 6h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_down_aligned = align_htf_to_ltf(prices, df_1d, donchian_down)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need 20-period calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_up_aligned[i]) or 
            np.isnan(donchian_down_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
        volume_ratio = volume_6h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: Donchian breakout + pivot direction + volume confirmation
            # Long: break above upper band AND pivot above price AND volume > 1.5x average
            if (close[i] > donchian_up_aligned[i] and 
                pivot_1w_aligned[i] > close[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: break below lower band AND pivot below price AND volume > 1.5x average
            elif (close[i] < donchian_down_aligned[i] and 
                  pivot_1w_aligned[i] < close[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian channel
            if close[i] < donchian_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian channel
            if close[i] > donchian_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Donchian_1wPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0