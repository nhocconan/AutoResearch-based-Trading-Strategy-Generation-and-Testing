# -*- coding: utf-8 -*-
#!/usr/bin/env python3
name = "6H_WeeklyPivot_DailyTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot levels from previous week
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    
    # Use previous week's data for current week's pivot levels
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    prev_close_w = df_1w['close'].shift(1).values
    
    # Calculate weekly pivot levels
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    r1_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 12.0
    s1_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 12.0
    r2_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 6.0
    s2_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 6.0
    r3_w = prev_close_w + (prev_high_w - prev_low_w) * 1.1 / 4.0
    s3_w = prev_close_w - (prev_high_w - prev_low_w) * 1.1 / 4.0
    
    # Align weekly pivot levels to 6h timeframe
    r1_w_6h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_6h = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_6h = align_htf_to_ltf(prices, df_1w, s2_w)
    r3_w_6h = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_w_6h = align_htf_to_ltf(prices, df_1w, s3_w)
    
    # Calculate daily EMA20 for trend filter
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average volume
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if weekly pivot levels not ready
        if np.isnan(r1_w_6h[i]) or np.isnan(s1_w_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 + volume confirmation + price above daily EMA20
            if close[i] > r2_w_6h[i] and volume_confirm[i] and close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S2 + volume confirmation + price below daily EMA20
            elif close[i] < s2_w_6h[i] and volume_confirm[i] and close[i] < ema20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2
            if close[i] < s2_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R2
            if close[i] > r2_w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals