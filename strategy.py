#!/usr/bin/env python3
# 6h_Donchian20_Breakout_WeeklyPivotDirection_Volume
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Long when weekly trend up (price above weekly pivot) and price breaks above 6h Donchian high(20) with volume > 1.5x average.
# Short when weekly trend down (price below weekly pivot) and price breaks below 6h Donchian low(20) with volume > 1.5x average.
# Uses weekly pivot for structural bias to avoid counter-trend trades, reducing whipsaw in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
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
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot point (P) = (H + L + C) / 3
    weekly_pivot = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        weekly_pivot[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: price above/below weekly pivot
        weekly_trend_up = close[i] > weekly_pivot_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + price breaks above Donchian high + volume confirmation
            if weekly_trend_up and close[i] > donchian_high[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + price breaks below Donchian low + volume confirmation
            elif not weekly_trend_up and close[i] < donchian_low[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly trend turns down or price breaks below Donchian low
            if not weekly_trend_up or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend turns up or price breaks above Donchian high
            if weekly_trend_up or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals