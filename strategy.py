# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_WeeklyPivot_Reversal_TrendFilter
Hypothesis: On daily timeframe, trade reversals from weekly pivot levels (S3/R3) with weekly trend filter (EMA50) and volume confirmation. This targets longer-term swings in both bull and bear markets by using weekly structure for direction and daily price action for entry. Expects 10-20 trades/year with low turnover to minimize fee drag.
"""

name = "1d_WeeklyPivot_Reversal_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly pivot points (using prior week)
    high_wk = df_1w['high'].shift(1).values
    low_wk = df_1w['low'].shift(1).values
    close_wk = df_1w['close'].shift(1).values
    
    pivot = (high_wk + low_wk + close_wk) / 3.0
    r1 = 2 * pivot - low_wk
    s1 = 2 * pivot - high_wk
    r2 = pivot + (high_wk - low_wk)
    s2 = pivot - (high_wk - low_wk)
    r3 = high_wk + 2 * (pivot - low_wk)
    s3 = low_wk - 2 * (high_wk - pivot)
    
    # Align weekly levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5x 20-day EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (50 for weekly EMA + 1 for shift)
    start_idx = 51
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs weekly EMA50
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price crosses below S3 then reverses up in uptrend with volume
            if i > 0 and close[i-1] <= s3_aligned[i-1] and close[i] > s3_aligned[i] and uptrend_1w and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses above R3 then reverses down in downtrend with volume
            elif i > 0 and close[i-1] >= r3_aligned[i-1] and close[i] < r3_aligned[i] and downtrend_1w and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 or trend fails
            if close[i] < s3_aligned[i] or not uptrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R3 or trend fails
            if close[i] > r3_aligned[i] or not downtrend_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals