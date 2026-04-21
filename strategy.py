#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_DonchianBreakout_VolumeFilter
Hypothesis: 6h timeframe with weekly pivot point direction filter and daily Donchian(20) breakouts.
Long when price breaks above Donchian high AND weekly pivot > prior week pivot (bullish bias).
Short when price breaks below Donchian low AND weekly pivot < prior week pivot (bearish bias).
Volume confirmation: current volume > 1.5x 24-period average (24*6h = 6 days).
Works in bull/bear by adapting to weekly pivot trend, avoiding counter-trend breaks.
Target: 12-37 trades/year (50-150 total over 4 years) to stay under 300 trade limit for 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: 20-day high/low
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_high = rolling_max(high_1d, 20)
    donchian_low = rolling_min(low_1d, 20)
    
    # Align Donchian levels to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load weekly data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly pivot trend: current pivot > prior week pivot = bullish bias
    pivot_bullish = weekly_pivot_aligned > np.roll(weekly_pivot_aligned, 1)
    pivot_bearish = weekly_pivot_aligned < np.roll(weekly_pivot_aligned, 1)
    # Handle first value
    pivot_bullish[0] = False
    pivot_bearish[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 24-period average (24*6h = 6 days)
        if i >= 24:
            vol_ma = prices['volume'].iloc[i-24:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: Donchian breakout up + weekly pivot bullish + volume
            if price > donchian_high_aligned[i] and pivot_bullish[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + weekly pivot bearish + volume
            elif price < donchian_low_aligned[i] and pivot_bearish[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low (reversal signal)
            if price < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high (reversal signal)
            if price > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_WeeklyPivot_DonchianBreakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0