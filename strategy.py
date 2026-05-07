#!/usr/bin/env python3
# 6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 6h chart strategy using weekly pivot point R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation. Weekly pivots provide robust support/resistance levels that work across bull and bear markets by capturing institutional interest at key weekly levels. The 1d EMA34 filter ensures trades align with the daily trend, while volume confirmation reduces false breakouts. Designed for low trade frequency (15-35/year) to minimize fee drag in bear markets like 2025.

name = "6h_WeeklyPivot_R1_S1_Breakout_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    weekly_pivot = (w_high + w_low + w_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - w_low
    weekly_s1 = 2 * weekly_pivot - w_high
    
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA34 on daily closes
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(4, 34)  # Ensure we have volume MA and EMA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close > weekly R1 with volume spike and daily uptrend
            if close[i] > weekly_r1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close < weekly S1 with volume spike and daily downtrend
            elif close[i] < weekly_s1_aligned[i] and volume[i] > 2.0 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch weekly S1 (opposite level) or trend failure
            if close[i] < weekly_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch weekly R1 (opposite level) or trend failure
            if close[i] > weekly_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals