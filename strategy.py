#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend Filter
Hypothesis: Weekly pivot levels (PP, R1, S1) from 1w data act as major support/resistance.
Breaking above weekly R1 with daily uptrend (price > EMA50) captures bullish momentum.
Breaking below weekly S1 with daily downtrend (price < EMA50) captures bearish momentum.
Designed for low trade frequency (~20-30/year) to minimize fee drag in 6h timeframe.
Works in both bull and bear markets by following the daily trend direction.
"""

name = "6h_WeeklyPivot_DailyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly Data for Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Standard pivot point calculation
    PP = (weekly_high + weekly_low + weekly_close) / 3.0
    R1 = 2 * PP - weekly_low
    S1 = 2 * PP - weekly_high
    
    # Shift to get previous week's levels
    PP_prev = np.roll(PP, 1)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    PP_prev[0] = np.nan
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    
    # Align to 6h timeframe
    PP_6h = align_htf_to_ltf(prices, df_1w, PP_prev)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1_prev)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1_prev)
    
    # === Daily EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or 
            np.isnan(ema_50_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R1 + price above daily EMA50 (uptrend)
            if close[i] > R1_6h[i] and close[i] > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S1 + price below daily EMA50 (downtrend)
            elif close[i] < S1_6h[i] and close[i] < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below weekly PP (mean reversion to pivot)
            if close[i] < PP_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly PP (mean reversion to pivot)
            if close[i] > PP_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals