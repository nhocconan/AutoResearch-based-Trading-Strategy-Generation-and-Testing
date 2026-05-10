#!/usr/bin/env python3
# 6h_WeeklyPivot_Reversal_1dTrend_Filter
# Hypothesis: Combine weekly pivot points with daily trend filter for reversal trading on 6h timeframe.
# Long when price rejects weekly S1/S2 with bullish rejection candle and daily uptrend.
# Short when price rejects weekly R1/R2 with bearish rejection candle and daily downtrend.
# Weekly pivots provide institutional levels; daily trend filter ensures trading with higher timeframe momentum.
# Designed for 50-150 total trades over 4 years to minimize fee drag.

name = "6h_WeeklyPivot_Reversal_1dTrend_Filter"
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
    open_price = prices['open'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for daily EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Rejection candle detection
        body_size = abs(close[i] - open_price[i])
        total_range = high[i] - low[i]
        upper_wick = high[i] - max(close[i], open_price[i])
        lower_wick = min(close[i], open_price[i]) - low[i]
        
        # Bullish rejection: long lower wick, small body, close near open
        bullish_rejection = (lower_wick > body_size * 2) and (body_size < total_range * 0.3)
        # Bearish rejection: long upper wick, small body, close near open
        bearish_rejection = (upper_wick > body_size * 2) and (body_size < total_range * 0.3)
        
        if position == 0:
            # Long entry: price near weekly S1/S2 with bullish rejection and daily uptrend
            near_s1 = abs(low[i] - s1_aligned[i]) < (pivot_aligned[i] * 0.005)  # Within 0.5% of S1
            near_s2 = abs(low[i] - s2_aligned[i]) < (pivot_aligned[i] * 0.005)  # Within 0.5% of S2
            if ((near_s1 or near_s2) and bullish_rejection and uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: price near weekly R1/R2 with bearish rejection and daily downtrend
            near_r1 = abs(high[i] - r1_aligned[i]) < (pivot_aligned[i] * 0.005)  # Within 0.5% of R1
            near_r2 = abs(high[i] - r2_aligned[i]) < (pivot_aligned[i] * 0.005)  # Within 0.5% of R2
            if ((near_r1 or near_r2) and bearish_rejection and downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches weekly pivot or rejection fails
            if close[i] >= pivot_aligned[i] or not bullish_rejection:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches weekly pivot or rejection fails
            if close[i] <= pivot_aligned[i] or not bearish_rejection:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals