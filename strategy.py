#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: On 1d timeframe, trade breakouts of weekly R1/S1 pivot levels with weekly EMA trend filter and volume spike confirmation. Weekly pivot levels provide strong institutional support/resistance, while the weekly EMA filter ensures alignment with the higher timeframe trend. Volume spike confirms institutional participation. This combination reduces false signals and works in both bull and bear markets by following the weekly trend. Target: 10-25 trades/year.
"""

name = "1d_WeeklyPivot_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for weekly pivot levels (calculated from previous week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    high_prev_week = df_1d['high'].shift(1).values  # Previous day's high (approximates weekly)
    low_prev_week = df_1d['low'].shift(1).values    # Previous day's low
    close_prev_week = df_1d['close'].shift(1).values # Previous day's close
    
    # Weekly pivot point calculation (standard formula)
    pivot_point = (high_prev_week + low_prev_week + close_prev_week) / 3.0
    r1 = 2 * pivot_point - low_prev_week
    s1 = 2 * pivot_point - high_prev_week
    
    # Align weekly pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get daily data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (50) and pivot points (need 1 day shift)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly EMA50
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: break above R1 in uptrend with volume spike
            if high[i] > r1_aligned[i] and uptrend_1w and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 in downtrend with volume spike
            elif low[i] < s1_aligned[i] and downtrend_1w and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or trend fails
            if low[i] < s1_aligned[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 or trend fails
            if high[i] > r1_aligned[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals