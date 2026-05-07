#!/usr/bin/env python3
# 1d_Pivot_Long_Short_Volume
# Hypothesis: Buy near weekly pivot support (S1) and sell near weekly pivot resistance (R1) on the daily chart, with volume confirmation and trend filter from weekly EMA. Designed to work in both bull and bear markets by fading extremes in weekly ranges. Target: 15-25 trades/year with strict entry conditions to minimize fee drag on 1d timeframe.

timeframe = "1d"
name = "1d_Pivot_Long_Short_Volume"
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
    
    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    pivot_point = (w_high + w_low + w_close) / 3.0
    w_range = w_high - w_low
    r1 = pivot_point + w_range
    s1 = pivot_point - w_range
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(w_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily volume spike detection: 2x average volume (20-period = ~1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near S1 (within 0.5%) with volume spike and above weekly EMA
            if (abs(close[i] - s1_aligned[i]) / s1_aligned[i] < 0.005 and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price near R1 (within 0.5%) with volume spike and below weekly EMA
            elif (abs(close[i] - r1_aligned[i]) / r1_aligned[i] < 0.005 and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches R1 or trend breaks
            if close[i] >= r1_aligned[i] or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches S1 or trend breaks
            if close[i] <= s1_aligned[i] or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals