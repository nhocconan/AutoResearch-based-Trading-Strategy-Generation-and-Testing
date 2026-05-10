#!/usr/bin/env python3
"""
1d_WeeklyPivot_TrendFollowing
Hypothesis: Use weekly pivot points (from 1w) as structural support/resistance.
In trending markets (determined by 1d EMA34), price tends to respect weekly pivot levels.
Long when price pulls back to weekly S1/S2 in uptrend with volume confirmation.
Short when price rallies to weekly R1/R2 in downtrend with volume confirmation.
Weekly pivots provide cleaner structure than daily pivots for 1d timeframe.
Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag.
Works in both bull (buy dips to support) and bear (sell rallies to resistance).
"""

name = "1d_WeeklyPivot_TrendFollowing"
timeframe = "1d"
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
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Weekly pivot points (from 1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points: P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 1d timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Need EMA34 and at least one weekly pivot
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or \
           np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or \
           np.isnan(s2_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume (SMA20)
        volume_confirm = volume[i] > 1.5 * vol_sma20_1d_aligned[i]
        
        if position == 0:
            # Long: Price near weekly S1/S2 in uptrend with volume confirmation
            # Allow 0.5% buffer around pivot levels
            near_s1 = abs(close[i] - s1_1w_aligned[i]) / s1_1w_aligned[i] < 0.005
            near_s2 = abs(close[i] - s2_1w_aligned[i]) / s2_1w_aligned[i] < 0.005
            if (near_s1 or near_s2) and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.30
                position = 1
            # Short: Price near weekly R1/R2 in downtrend with volume confirmation
            elif (abs(close[i] - r1_1w_aligned[i]) / r1_1w_aligned[i] < 0.005 or
                  abs(close[i] - r2_1w_aligned[i]) / r2_1w_aligned[i] < 0.005) and \
                 close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Price reaches weekly pivot or R1, or trend reversal
            if (close[i] >= pivot_1w_aligned[i] or 
                close[i] >= r1_1w_aligned[i] or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price reaches weekly pivot or S1, or trend reversal
            if (close[i] <= pivot_1w_aligned[i] or 
                close[i] <= s1_1w_aligned[i] or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals