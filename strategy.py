#!/usr/bin/env python3
"""
6h_WeeklyPivot_Pullback_1dTrend
Hypothesis: In trending markets (1d EMA34), price pulls back to weekly pivot support/resistance before continuing the trend.
Long when price pulls back to weekly S1 in uptrend, short when price pulls back to weekly R1 in downtrend.
Weekly pivots provide strong structural levels; pullbacks offer high-probability entries with tight stops.
Targets 80-120 trades over 4 years (20-30/year) to minimize fee drag.
Works in bull (buy dips to support) and bear (sell rallies to resistance).
"""

name = "6h_WeeklyPivot_Pullback_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Weekly pivot points (from 1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Need EMA34 and at least one weekly pivot
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price pulls back to weekly S1 in uptrend
            # Allow 0.3% buffer around S1 for entry
            near_s1 = abs(close[i] - s1_1w_aligned[i]) / s1_1w_aligned[i] < 0.003
            if near_s1 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price pulls back to weekly R1 in downtrend
            elif abs(close[i] - r1_1w_aligned[i]) / r1_1w_aligned[i] < 0.003 and \
                 close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price reaches weekly pivot or R1, or trend reversal
            if (close[i] >= pivot_1w_aligned[i] or 
                close[i] >= r1_1w_aligned[i] or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price reaches weekly pivot or S1, or trend reversal
            if (close[i] <= pivot_1w_aligned[i] or 
                close[i] <= s1_1w_aligned[i] or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals