#!/usr/bin/env python3
# 6h_Pivot_Reversion_1dTrend_Volume
# Hypothesis: 6h chart strategy using daily pivot points for mean reversion with 1w trend filter and volume confirmation. 
# Mean reversion works well in ranging markets (2023-2024) while trend filter avoids counter-trend trades in strong trends (2021, 2025).
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee drag.

timeframe = "6h"
name = "6h_Pivot_Reversion_1dTrend_Volume"
leverage = 1.0

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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily pivot points: P = (H+L+C)/3
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    pivot = (d_high + d_low + d_close) / 3.0
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily pivot and weekly EMA to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection: 1.5x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(4, 50)  # Ensure we have volume MA and weekly EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below pivot with volume spike and weekly uptrend
            if close[i] < pivot_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above pivot with volume spike and weekly downtrend
            elif close[i] > pivot_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses above pivot or trend failure
            if close[i] > pivot_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses below pivot or trend failure
            if close[i] < pivot_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals