#!/usr/bin/env python3
"""
12h_WeeklyPivot_R1S1_Breakout_Volume
Hypothesis: Buy at weekly pivot S1 with volume confirmation in uptrend, sell at R1 in downtrend.
Uses 1d pivot levels from previous week, volume > 1.5x 24-period average, and 1d EMA50 trend filter.
Designed for low turnover (<20 trades/year) to minimize fee drag while capturing mean reversion
within the weekly pivot range in ranging markets and trend continuation in trending markets.
Works in bull markets via buying dips to S1 in uptrend, in bear markets via selling rallies to R1 in downtrend.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly pivot points from 1d data (using prior week's high, low, close)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot: need to group daily data into weeks
    # We'll use the prior week's H/L/C for current week's pivot
    # For simplicity, use rolling window of 5 trading days (1 week)
    if len(high_1d) >= 5:
        weekly_high = np.full(len(high_1d), np.nan)
        weekly_low = np.full(len(high_1d), np.nan)
        weekly_close = np.full(len(high_1d), np.nan)
        
        for i in range(5, len(high_1d)):
            weekly_high[i] = np.max(high_1d[i-5:i])
            weekly_low[i] = np.min(low_1d[i-5:i])
            weekly_close[i] = close_1d[i-1]  # Previous day's close as weekly close proxy
        
        # Pivot point = (H + L + C) / 3
        pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
        # Support 1 = (2 * P) - H
        s1 = (2 * pivot_point) - weekly_high
        # Resistance 1 = (2 * P) - L
        r1 = (2 * pivot_point) - weekly_low
    else:
        # Not enough data
        pivot_point = np.full(len(high_1d), np.nan)
        s1 = np.full(len(high_1d), np.nan)
        r1 = np.full(len(high_1d), np.nan)
    
    # Align weekly pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume filter: current volume > 1.5 x 24-period average (24 * 12h = 12 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    # 1d EMA50 trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    k = 2 / (50 + 1)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[0:51])
        else:
            ema50_1d[i] = close_1d[i] * k + ema50_1d[i-1] * (1 - k)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50)  # Ensure volume MA and EMA ready
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at or below S1 with volume confirmation and 1d uptrend
            if (close[i] <= s1_aligned[i] and vol_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price at or above R1 with volume confirmation and 1d downtrend
            elif (close[i] >= r1_aligned[i] and vol_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 or trend turns down
            if (close[i] >= r1_aligned[i] or close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 or trend turns up
            if (close[i] <= s1_aligned[i] or close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0