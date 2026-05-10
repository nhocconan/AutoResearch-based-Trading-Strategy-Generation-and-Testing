#!/usr/bin/env python3
"""
4h_Weekly_Pivot_Momentum
Hypothesis: Price reacts strongly to weekly pivot levels (S3/S2/R2/R3) with volume confirmation and trend alignment. Works in bull markets via buying S3/S2 bounces and in bear markets via selling R2/R3 rejections. Uses weekly pivots for structure and 4h EMA50 for trend filter to avoid counter-trend trades. Target: 20-40 trades/year to minimize fee drag.
"""

name = "4h_Weekly_Pivot_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # S1 = 2*P - H, S2 = P - (H - L), S3 = L - 2*(H - P)
    # R1 = 2*P - L, R2 = P + (H - L), R3 = H + 2*(P - L)
    pivot_w = (high_w + low_w + close_w) / 3.0
    s1_w = 2 * pivot_w - high_w
    s2_w = pivot_w - (high_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r1_w = 2 * pivot_w - low_w
    r2_w = pivot_w + (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    
    # Align weekly pivots to 4h timeframe (with 1-bar delay for completed weekly bar)
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_w, s3_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_w, r3_w)
    
    # Get 4h data for trend filter and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h data for signal generation
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (5 weeks) and 4h EMA50 (50)
    start_idx = max(5, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(pivot_w_aligned[i]) or
            np.isnan(s3_w_aligned[i]) or
            np.isnan(r3_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 4h EMA50
        uptrend_4h = close[i] > ema50_4h_aligned[i]
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        # Volume filter: current 4h volume > 1.8x 20-period average
        vol_ma20 = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
        vol_ma20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma20)
        volume_filter = volume[i] > vol_ma20_aligned[i] * 1.8
        
        # Distance to weekly levels (as fraction of price)
        dist_to_s3 = (close[i] - s3_w_aligned[i]) / close[i] if s3_w_aligned[i] > 0 else 1.0
        dist_to_r3 = (r3_w_aligned[i] - close[i]) / close[i] if r3_w_aligned[i] > 0 else 1.0
        
        # Entry zones: within 0.5% of weekly S3 or R3
        near_s3 = abs(dist_to_s3) < 0.005
        near_r3 = abs(dist_to_r3) < 0.005
        
        if position == 0:
            # Long entry: near S3 + uptrend + volume
            if near_s3 and uptrend_4h and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: near R3 + downtrend + volume
            elif near_r3 and downtrend_4h and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches S2 or trend fails
            if close[i] <= s2_w_aligned[i] or not uptrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches R2 or trend fails
            if close[i] >= r2_w_aligned[i] or not downtrend_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals