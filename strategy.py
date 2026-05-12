#!/usr/bin/env python3
name = "6h_WeeklyPivot_Fade_Breakout"
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
    
    # === 1W DATA FOR PIVOT POINTS ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivot levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w = r3_1w + (r2_1w - r1_1w)  # R4 = R3 + (R2 - R1)
    s4_1w = s3_1w - (s1_1w - s2_1w)  # S4 = S3 - (S1 - S2)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 1D DATA FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # FADE AT R3/S3: Price touches R3/S3 with volume spike, expect reversal
            # LONG at S3: price <= S3 and bouncing up
            if (close[i] <= s3_1w_aligned[i] and 
                close[i] > s3_1w_aligned[i] * 0.998 and  # Within 0.2% of S3
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT at R3: price >= R3 and rejecting down
            elif (close[i] >= r3_1w_aligned[i] and 
                  close[i] < r3_1w_aligned[i] * 1.002 and  # Within 0.2% of R3
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            # BREAKOUT CONTINUATION AT R4/S4: Price breaks R4/S4 with volume spike
            # LONG on R4 breakout: price > R4
            elif (close[i] > r4_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT on S4 breakdown: price < S4
            elif (close[i] < s4_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price reaches R3 (take profit) or breaks below S1 (stop)
            if (close[i] >= r3_1w_aligned[i] or 
                close[i] <= s1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S3 (take profit) or breaks above R1 (stop)
            if (close[i] <= s3_1w_aligned[i] or 
                close[i] >= r1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals