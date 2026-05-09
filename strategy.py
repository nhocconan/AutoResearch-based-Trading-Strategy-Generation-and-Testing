#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (HIGH LEVEL STRUCTURE)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    H = df_1w['high'].values
    L = df_1w['low'].values
    C = df_1w['close'].values
    
    pivot = (H + L + C) / 3
    R1 = 2 * pivot - L
    S1 = 2 * pivot - H
    R2 = pivot + (H - L)
    S2 = pivot - (H - L)
    R3 = H + 2 * (pivot - L)
    S3 = L - 2 * (H - pivot)
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Get daily trend filter (more responsive than weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 2.0 * 20-period average (adjust for 6h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for volume MA and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        piv = pivot_aligned[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        r2 = R2_aligned[i]
        s2 = S2_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        trend = ema50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above weekly R3 with volume and above daily trend
            if close[i] > r3 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly S3 with volume and below daily trend
            elif close[i] < s3 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to weekly pivot (mean reversion to key level)
            if close[i] < piv:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot (mean reversion to key level)
            if close[i] > piv:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals