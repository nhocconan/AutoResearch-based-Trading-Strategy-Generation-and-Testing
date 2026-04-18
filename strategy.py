#!/usr/bin/env python3
"""
6h_WeeklyPivot_R3S3_Fade_R4S4_Breakout_Volume_v1
Hypothesis: Use 1-week Pivot Points (R3, S3, R4, S4) with volume confirmation to capture mean-reversion at extreme weekly levels (R3/S3) and breakout continuation at stronger levels (R4/S4). Works in both bull/bear markets by fading extremes and riding breakouts with volume confirmation. Targets 15-25 trades/year on 6H timeframe.
"""

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
    volume = prices['volume'].values
    
    # Calculate Weekly Pivot Points from previous week (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Use previous week's data for pivot calculation (avoid look-ahead)
    prev_high = high_1w[:-1]  # All except last (current forming) week
    prev_low = low_1w[:-1]
    prev_close = close_1w[:-1]
    
    # Pivot Point calculation
    pp = (prev_high + prev_low + prev_close) / 3.0
    # R3, S3, R4, S4 levels
    r3 = pp + 2 * (prev_high - prev_low)
    s3 = pp - 2 * (prev_high - prev_low)
    r4 = pp + 3 * (prev_high - prev_low)
    s4 = pp - 3 * (prev_high - prev_low)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w[:-1], r3)  # Use df_1w[:-1] to match prev data
    s3_aligned = align_htf_to_ltf(prices, df_1w[:-1], s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w[:-1], r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w[:-1], s4)
    
    # Volume confirmation: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Fade at R3/S3: Short at R3 with volume, Long at S3 with volume
            if price > r3_val and vol_spike:
                signals[i] = -0.25
                position = -1
            elif price < s3_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Breakout continuation at R4/S4: Long at R4 break, Short at S4 break
            elif price > r4_val and vol_spike:
                signals[i] = 0.25
                position = 1
            elif price < s4_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            signals[i] = 0.25
            # Exit conditions: 
            # 1. Mean reversion: price returns to pivot area (between S3 and R3)
            # 2. Opposite extreme touched
            if (price >= s3_val and price <= r3_val) or price < s4_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:  # Short position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Mean reversion: price returns to pivot area (between S3 and R3)
            # 2. Opposite extreme touched
            if (price >= s3_val and price <= r3_val) or price > r4_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_R3S3_Fade_R4S4_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0