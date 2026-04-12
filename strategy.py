#!/usr/bin/env python3
"""
6h_1d_wkly_pivot_volume_fade_v1
Hypothesis: 6-hour strategy using 1-week pivot points for trend direction and 1-day pivot levels for mean-reversion entries, with volume confirmation.
Works in bull/bear by fading at weekly S1/R1 and S2/R2 when price deviates from the weekly pivot, using 1-day pivots as entry triggers and volume to confirm reversals.
Targets 15-35 trades per year (60-140 total over 4 years) to minimize fee drag.
"""

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
    
    # Get 1d data for entry pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range for daily pivots
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    range_1d = prev_high_1d - prev_low_1d
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r1_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s1_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    r2_1d = pivot_1d + 2 * (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - 2 * (prev_high_1d - prev_low_1d)
    
    # Get 1w data for trend pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous 1w bar's range for weekly pivots
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    range_1w = prev_high_1w - prev_low_1w
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    r1_1w = pivot_1w + (prev_high_1w - prev_low_1w)
    s1_1w = pivot_1w - (prev_high_1w - prev_low_1w)
    r2_1w = pivot_1w + 2 * (prev_high_1w - prev_low_1w)
    s2_1w = pivot_1w - 2 * (prev_high_1w - prev_low_1w)
    
    # Align daily pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price near weekly S1/S2 AND below daily pivot with volume confirmation
        if (close[i] <= s1_1w_aligned[i] * 1.02 and close[i] >= s2_1w_aligned[i] * 0.98 and
            close[i] < pivot_1d_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price near weekly R1/R2 AND above daily pivot with volume confirmation
        elif (close[i] >= r1_1w_aligned[i] * 0.98 and close[i] <= r2_1w_aligned[i] * 1.02 and
              close[i] > pivot_1d_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses weekly pivot in opposite direction
        elif position == 1 and close[i] > pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] < pivot_1w_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_wkly_pivot_volume_fade_v1"
timeframe = "6h"
leverage = 1.0