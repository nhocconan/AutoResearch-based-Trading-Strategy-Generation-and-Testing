#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Daily Trend Filter
- Uses weekly pivot points (PP, R1, S1, R2, S2) calculated from prior week
- Long when price bounces off weekly S1/S2 with daily uptrend
- Short when price bounces off weekly R1/R2 with daily downtrend
- Uses daily EMA50 as trend filter to avoid counter-trend trades
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Reversal_DailyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points using prior week's data
    # PP = (H+L+C)/3, R1 = 2*PP-L, S1 = 2*PP-H, R2 = PP+(H-L), S2 = PP-(H-L)
    n_weekly = len(close_weekly)
    pivot_PP = np.full(n_weekly, np.nan)
    pivot_S1 = np.full(n_weekly, np.nan)
    pivot_R1 = np.full(n_weekly, np.nan)
    pivot_S2 = np.full(n_weekly, np.nan)
    pivot_R2 = np.full(n_weekly, np.nan)
    
    for i in range(1, n_weekly):
        H = high_weekly[i-1]
        L = low_weekly[i-1]
        C = close_weekly[i-1]
        PP = (H + L + C) / 3.0
        pivot_PP[i] = PP
        pivot_S1[i] = 2 * PP - H
        pivot_R1[i] = 2 * PP - L
        pivot_S2[i] = PP - (H - L)
        pivot_R2[i] = PP + (H - L)
    
    # Align weekly pivots to 6h timeframe
    pivot_PP_aligned = align_htf_to_ltf(prices, df_weekly, pivot_PP)
    pivot_S1_aligned = align_htf_to_ltf(prices, df_weekly, pivot_S1)
    pivot_R1_aligned = align_htf_to_ltf(prices, df_weekly, pivot_R1)
    pivot_S2_aligned = align_htf_to_ltf(prices, df_weekly, pivot_S2)
    pivot_R2_aligned = align_htf_to_ltf(prices, df_weekly, pivot_R2)
    
    # Daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 5:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    # Daily EMA50 for trend filter
    ema_50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_PP_aligned[i]) or np.isnan(pivot_S1_aligned[i]) or 
            np.isnan(pivot_R1_aligned[i]) or np.isnan(pivot_S2_aligned[i]) or 
            np.isnan(pivot_R2_aligned[i]) or np.isnan(ema_50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or crosses above S1/S2 with daily uptrend
            long_cond = ((close[i] >= pivot_S1_aligned[i] or close[i] >= pivot_S2_aligned[i]) and
                        ema_50_daily_aligned[i] > ema_50_daily_aligned[i-1])
            
            # Short: price touches or crosses below R1/R2 with daily downtrend
            short_cond = ((close[i] <= pivot_R1_aligned[i] or close[i] <= pivot_R2_aligned[i]) and
                         ema_50_daily_aligned[i] < ema_50_daily_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches or crosses below pivot point
            if close[i] <= pivot_PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches or crosses above pivot point
            if close[i] >= pivot_PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals