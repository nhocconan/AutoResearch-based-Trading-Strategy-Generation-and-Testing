#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_TrendFilter
Hypothesis: Use weekly pivot levels (R3/S3) as support/resistance and Donchian(20) breakout on 6h for entry, filtered by daily trend (price > EMA50). Go long when price breaks above Donchian high with price above weekly R3 and daily EMA50, short when price breaks below Donchian low with price below weekly S3 and daily EMA50. Weekly pivots provide strong institutional levels, Donchian captures breakouts, daily EMA50 filters counter-trend trades. Designed for 6h timeframe to target 15-35 trades/year.
"""

name = "6h_WeeklyPivot_DonchianBreakout_TrendFilter"
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
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly data for pivot points (R3, S3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r3_1w = high_1w + 2.0 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2.0 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Donchian breakout above weekly R3 + daily uptrend
            if high[i] > high_max_20[i] and close[i] > r3_1w_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakout below weekly S3 + daily downtrend
            elif low[i] < low_min_20[i] and close[i] < s3_1w_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or weekly S3
            if low[i] < low_min_20[i] or close[i] < s3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or weekly R3
            if high[i] > high_max_20[i] or close[i] > r3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals