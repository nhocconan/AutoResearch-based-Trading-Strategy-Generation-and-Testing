#!/usr/bin/env python3
# 1d_WeeklyPivot_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Daily chart strategy using Weekly (R1/S1) pivot breakouts with 1-week EMA50 trend filter and volume confirmation. Designed for low trade frequency (7-25/year) to minimize fee drag, with trend filter to work in both bull and bear markets. Target: 30-100 total trades over 4 years.

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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w closes
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get weekly data for Weekly Pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) == 0:
        return np.zeros(n)
    
    # Calculate Weekly Pivot R1 and S1 from weekly high/low/close
    w_high = df_w['high'].values
    w_low = df_w['low'].values
    w_close = df_w['close'].values
    
    pivot = (w_high + w_low + w_close) / 3.0
    weekly_r1 = 2 * pivot - w_low
    weekly_s1 = 2 * pivot - w_high
    
    weekly_r1_aligned = align_htf_to_ltf(prices, df_w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_w, weekly_s1)
    
    # Volume spike detection: 2x average volume (20-period = 20 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > Weekly R1 with volume spike and 1w uptrend
            if close[i] > weekly_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < Weekly S1 with volume spike and 1w downtrend
            elif close[i] < weekly_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch Weekly S1 (opposite level) or trend failure
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch Weekly R1 (opposite level) or trend failure
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals