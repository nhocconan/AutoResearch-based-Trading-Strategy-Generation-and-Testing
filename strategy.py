#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_1dTrend_Volume
Hypothesis: Weekly pivot breakout (R4/S4) with daily EMA50 trend filter and volume spike.
Uses weekly pivots for structural support/resistance and daily trend to filter direction.
Works in both bull/bear markets by only taking breakouts in trend direction.
Target: 15-25 trades/year to avoid fee drag.
"""

name = "6h_Weekly_Pivot_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivots and daily data for trend
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 1 or len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot points from previous week
    high_prev_w = df_1w['high'].values
    low_prev_w = df_1w['low'].values
    close_prev_w = df_1w['close'].values
    
    # Standard pivot point formulas
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # R2 = P + (H - L)
    # S2 = P - (H - L)
    # R3 = H + 2*(P - L)
    # S3 = L - 2*(H - P)
    # R4 = R3 + (H - L)
    # S4 = S3 - (H - L)
    pivot = (high_prev_w + low_prev_w + close_prev_w) / 3
    r1 = 2 * pivot - low_prev_w
    s1 = 2 * pivot - high_prev_w
    r2 = pivot + (high_prev_w - low_prev_w)
    s2 = pivot - (high_prev_w - low_prev_w)
    r3 = high_prev_w + 2 * (pivot - low_prev_w)
    s3 = low_prev_w - 2 * (high_prev_w - pivot)
    r4 = r3 + (high_prev_w - low_prev_w)
    s4 = s3 - (high_prev_w - low_prev_w)
    
    # Align weekly pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Get price, volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R4 with uptrend and volume
            if close[i] > r4_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S4 with downtrend and volume
            elif close[i] < s4_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below S4 or trend change
            if close[i] < s4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above R4 or trend change
            if close[i] > r4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals