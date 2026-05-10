#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: Price breaks Donchian(20) channel on 6h timeframe with weekly pivot bias and volume confirmation.
Weekly pivot provides directional bias: long only above weekly pivot, short only below.
Volume confirms breakout strength. Works in bull/bear by filtering with weekly pivot direction.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot and bias
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Donchian channel (20-period) on 6h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Weekly average volume for confirmation
    vol_1w = df_1w['volume'].values
    vol_avg_1w = np.full(len(vol_1w), np.nan)
    if len(vol_1w) >= 20:
        for i in range(19, len(vol_1w)):
            vol_avg_1w[i] = np.mean(vol_1w[i-19:i+1])
    
    # Align weekly data to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 20)  # Wait for Donchian and volume average
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(vol_avg_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average weekly volume (scaled)
        # Approx: 28 six-hour bars in a week (7 days * 4 bars/day)
        vol_1w_scaled = vol_avg_1w_aligned[i] / 28.0
        volume_confirm = volume[i] > 2.0 * vol_1w_scaled  # Stricter for lower frequency
        
        # Price relative to Donchian channels
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Weekly pivot bias
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        if position == 0:
            # Long: breakout up, above weekly pivot, with volume
            if breakout_up and above_pivot and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout down, below weekly pivot, with volume
            elif breakout_down and below_pivot and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price retests lower Donchian band or pivot loses bias
            if close[i] < lowest_low[i] or not above_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price retests upper Donchian band or pivot loses bias
            if close[i] > highest_high[i] or not below_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals