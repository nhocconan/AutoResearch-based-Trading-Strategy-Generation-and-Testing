#!/usr/bin/env python3
# 6h_1d_pivot_reversion_with_volume
# Hypothesis: 6-hour price reversals from daily pivot points with volume confirmation
# Works in bull/bear by fading extreme deviations from daily pivot (mean reversion)
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag

name = "6h_1d_pivot_reversion_with_volume"
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's data for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Standard pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Support and resistance levels
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.3x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price touches S1/S2 with volume confirmation
        long_signal = False
        if (low[i] <= s1_aligned[i] or low[i] <= s2_aligned[i]) and vol_confirm[i]:
            long_signal = True
        
        # Short entry: price touches R1/R2 with volume confirmation
        short_signal = False
        if (high[i] >= r1_aligned[i] or high[i] >= r2_aligned[i]) and vol_confirm[i]:
            short_signal = True
        
        # Exit conditions
        exit_long = position == 1 and (high[i] >= pivot_aligned[i] or low[i] <= s2_aligned[i])
        exit_short = position == -1 and (low[i] <= pivot_aligned[i] or high[i] >= r2_aligned[i])
        
        # Update position
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
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