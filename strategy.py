#!/usr/bin/env python3
# 1d_WeeklyPivot_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Daily chart strategy using weekly pivot R1/S1 breakouts with weekly EMA50 trend filter and volume confirmation. Designed for low trade frequency (10-20/year) to minimize fee drag, with trend filter to work in both bull and bear markets. Target: 40-80 total trades over 4 years.

timeframe = "1d"
name = "1d_WeeklyPivot_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly closes
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get weekly data for pivot points
    df_1w_pivot = get_htf_data(prices, '1w')
    if len(df_1w_pivot) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    w_high = df_1w_pivot['high'].values
    w_low = df_1w_pivot['low'].values
    w_close = df_1w_pivot['close'].values
    
    pivot_p = (w_high + w_low + w_close) / 3.0
    pivot_r1 = 2 * pivot_p - w_low
    pivot_s1 = 2 * pivot_p - w_high
    
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1w_pivot, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1w_pivot, pivot_s1)
    
    # Volume spike detection: 2x average volume (20-period = ~1 month on daily chart)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(pivot_r1_aligned[i]) or np.isnan(pivot_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > R1 with volume spike and weekly uptrend
            if close[i] > pivot_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < S1 with volume spike and weekly downtrend
            elif close[i] < pivot_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure
            if close[i] < pivot_s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure
            if close[i] > pivot_r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals