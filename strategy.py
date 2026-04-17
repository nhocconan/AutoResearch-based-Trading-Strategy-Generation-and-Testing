#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1S1_FadeWithTrend
Fade at weekly pivot R1/S1 with trend filter from 1d EMA200.
Long: price < weekly S1 and price > 1d EMA200 (buy dip in uptrend).
Short: price > weekly R1 and price < 1d EMA200 (sell rally in downtrend).
Exit when price crosses 1d EMA200 or reaches opposite pivot level (R2/S2).
Targets 15-30 trades/year (~60-120 total over 4 years).
"""

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
    
    # === Weekly pivot points (using prior week's OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivots from prior week's data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot_point = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot_point - weekly_low
    s1 = 2 * pivot_point - weekly_high
    r2 = pivot_point + (weekly_high - weekly_low)
    s2 = pivot_point - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # === 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isclose(pivot_point_aligned[i], 0) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price below S1 (support) AND above 1d EMA200 (uptrend filter)
            if (low[i] <= s1_aligned[i] and 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price above R1 (resistance) AND below 1d EMA200 (downtrend filter)
            elif (high[i] >= r1_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses above EMA200 OR reaches R2 (next resistance)
            if (close[i] >= ema_200_1d_aligned[i] or 
                high[i] >= r2_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses below EMA200 OR reaches S2 (next support)
            if (close[i] <= ema_200_1d_aligned[i] or 
                low[i] <= s2_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1S1_FadeWithTrend"
timeframe = "6h"
leverage = 1.0