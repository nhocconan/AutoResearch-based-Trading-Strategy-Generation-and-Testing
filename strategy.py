#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeFilter
Hypothesis: Uses 6h Donchian(20) breakouts with weekly pivot direction (from weekly pivot points) and volume confirmation to capture strong trend moves. Weekly pivot provides directional bias from higher timeframe structure, reducing false breakouts in chop. Designed for 15-35 trades/year to minimize fee drag while capturing major moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Get daily data for volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    weekly_pivot = np.zeros_like(close_weekly)
    weekly_r1 = np.zeros_like(close_weekly)
    weekly_s1 = np.zeros_like(close_weekly)
    
    for i in range(len(close_weekly)):
        if i == 0:
            weekly_pivot[i] = close_weekly[i]
            weekly_r1[i] = close_weekly[i]
            weekly_s1[i] = close_weekly[i]
        else:
            pivot = (high_weekly[i-1] + low_weekly[i-1] + close_weekly[i-1]) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - low_weekly[i-1]
            weekly_s1[i] = 2 * pivot - high_weekly[i-1]
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Calculate 6h Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and MA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above weekly pivot
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and below weekly pivot
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < weekly_pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below weekly pivot or Donchian low
            if close[i] < weekly_pivot_aligned[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above weekly pivot or Donchian high
            if close[i] > weekly_pivot_aligned[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeFilter"
timeframe = "6h"
leverage = 1.0