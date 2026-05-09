#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_VolumeFilter
# Hypothesis: Weekly pivot-based Donchian breakout with volume confirmation.
# Weekly pivot defines long-term structure; Donchian breakout captures momentum; volume filter confirms institutional participation.
# Works in bull/bear: Long only when price above weekly pivot (bullish bias), short only when below (bearish bias).
# Uses 60-bar Donchian channels for breakout detection and 20-bar volume average for confirmation.

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - low_1w
    s1 = 2 * weekly_pivot - high_1w
    r2 = weekly_pivot + (high_1w - low_1w)
    s2 = weekly_pivot - (high_1w - low_1w)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Calculate 60-bar Donchian channels (breakout detection)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    if len(high) >= 60:
        for i in range(60, len(high)):
            donchian_high[i] = np.max(high[i-60:i])
            donchian_low[i] = np.min(low[i-60:i])
    
    # Volume confirmation: current volume / 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma > 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above weekly pivot AND breaks above Donchian high AND volume spike
            if (close[i] > weekly_pivot_aligned[i] and 
                high[i] > donchian_high[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly pivot AND breaks below Donchian low AND volume spike
            elif (close[i] < weekly_pivot_aligned[i] and 
                  low[i] < donchian_low[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR falls below weekly pivot
            if low[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR rises above weekly pivot
            if high[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals