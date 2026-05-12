#!/usr/bin/env python3
# 6h_Pivot_Reversion_Volume
# Hypothesis: Mean reversion from daily pivot points (PP, R1, S1) on 6b timeframe with volume confirmation.
# In ranging markets, price tends to revert to the mean (pivot point) after reaching support/resistance levels (S1/R1).
# Volume confirmation ensures the reversion has momentum. Works in both bull and bear markets by fading extremes.

name = "6h_Pivot_Reversion_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_to_ltf, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Points (Standard Formula) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate pivot points from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    r1 = (2 * pp) - low_1d
    # Support 1 (S1) = (2 * PP) - High
    s1 = (2 * pp) - high_1d
    
    # Align to 6h timeframe (use previous day's pivot for current day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price touches or goes below S1 with volume, expect reversion to PP
            if close[i] <= s1_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R1 with volume, expect reversion to PP
            elif close[i] >= r1_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches or crosses PP (mean reversion complete)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches or crosses PP (mean reversion complete)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals