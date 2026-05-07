#!/usr/bin/env python3
# 6h_Donchian20_Breakout_WeeklyPivotDirection_Volume
# Hypothesis: Donchian(20) breakout on 6h timeframe filtered by weekly pivot direction (from 1w) and volume confirmation.
# Weekly pivot provides institutional bias - long when price above weekly pivot, short when below.
# Donchian breakout captures momentum, volume confirms strength. Works in bull/bear via pivot filter.
# Target: 15-35 trades/year (60-140 total over 4 years) to avoid fee drag.

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    
    # Shift by 1 to use previous week's levels (no look-ahead)
    pivot_w = np.roll(pivot_w, 1)
    r1_w = np.roll(r1_w, 1)
    s1_w = np.roll(s1_w, 1)
    pivot_w[0] = np.nan
    r1_w[0] = np.nan
    s1_w[0] = np.nan
    
    # Align weekly levels to 6h timeframe
    pivot_w_6h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_w_6h[i]) or np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly pivot + volume spike
            if close[i] > donchian_high[i] and close[i] > pivot_w_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly pivot + volume spike
            elif close[i] < donchian_low[i] and close[i] < pivot_w_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below Donchian low or below weekly pivot
            if close[i] < donchian_low[i] or close[i] < pivot_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above Donchian high or above weekly pivot
            if close[i] > donchian_high[i] or close[i] > pivot_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals