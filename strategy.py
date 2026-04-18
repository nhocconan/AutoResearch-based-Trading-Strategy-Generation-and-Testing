#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Fade_R4S4_Breakout_Volume_v1
Hypothesis: On 6h timeframe, fade price at weekly pivot R3/S3 levels (mean reversion in ranging markets) and breakout continuation at R4/S4 levels (trend continuation in strong moves), both confirmed by volume spikes. Weekly pivots provide strong institutional levels that work in both bull and bear markets by capturing mean reversion at extremes and trend acceleration at breakouts. Target: 20-40 trades/year to minimize fee drag.
"""

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
    
    # Calculate weekly pivot points from previous week (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly support/resistance levels
    r3_1w = pivot_1w + range_1w * 1.1
    s3_1w = pivot_1w - range_1w * 1.1
    r4_1w = pivot_1w + range_1w * 1.5
    s4_1w = pivot_1w - range_1w * 1.5
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation: >1.8x 30-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot = pivot_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Fade at R3/S3: mean reversion when price reaches extreme weekly levels
            if price >= r3 and vol_spike:
                signals[i] = -0.25  # Short at R3
                position = -1
            elif price <= s3 and vol_spike:
                signals[i] = 0.25   # Long at S3
                position = 1
            # Breakout continuation at R4/S4: trend acceleration when price breaks weekly extremes
            elif price > r4 and vol_spike:
                signals[i] = 0.25   # Long breakout above R4
                position = 1
            elif price < s4 and vol_spike:
                signals[i] = -0.25  # Short breakdown below S4
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit long: price returns to pivot (mean reversion) or breaks S4 (stop)
            if price <= pivot or price < s4:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit short: price returns to pivot (mean reversion) or breaks R4 (stop)
            if price >= pivot or price > r4:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Fade_R4S4_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0