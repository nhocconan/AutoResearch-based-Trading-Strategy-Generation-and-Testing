#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_VolumeFilter
# Hypothesis: Combines weekly pivot direction (from 1w), Donchian breakout (20-period) on 6h, and volume spike filter.
# Weekly pivot provides long-term bias (bull/bear), Donchian captures breakouts in direction of trend, volume confirms institutional participation.
# Works in both bull and bear markets: weekly pivot keeps us on correct side of major trend, avoiding counter-trend whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entry conditions.

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter"
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
    
    # Get weekly data for pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Weekly pivot and support/resistance levels
    pivot = (ph + pl + pc) / 3
    r1 = 2 * pivot - pl
    s1 = 2 * pivot - ph
    r2 = pivot + (ph - pl)
    s2 = pivot - (ph - pl)
    r3 = ph + 2 * (pivot - pl)
    s3 = pl - 2 * (ph - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback - 1, len(high)):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 19)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine trend from weekly pivot: above pivot = bullish, below = bearish
            is_bullish = close[i] > pivot_aligned[i]
            
            # Enter long: bullish weekly bias AND price breaks above Donchian high AND volume spike
            if is_bullish and high[i] > highest_high[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish weekly bias AND price breaks below Donchian low AND volume spike
            elif not is_bullish and low[i] < lowest_low[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly bias turns bearish
            if low[i] < lowest_low[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly bias turns bullish
            if high[i] > highest_high[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals