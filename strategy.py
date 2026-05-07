#!/usr/bin/env python3
name = "6h_TurtleSoup_Reversal_WeeklyPivot_HTFTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (weekly close from Monday 00:00 UTC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader pivots)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for trend filter (1d EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_ok = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Need 30 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Turtle Soup Long: Fake breakdown below S1, then reverse above S1
            # Condition: price was below S1 in previous bar, now back above S1
            if low[i] < s1_aligned[i] and close[i] > s1_aligned[i]:
                # Additional confirmation: price above weekly pivot and above daily EMA50
                if close[i] > pivot_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_ok[i]:
                    signals[i] = 0.25
                    position = 1
            # Turtle Soup Short: Fake breakout above R1, then reverse below R1
            elif high[i] > r1_aligned[i] and close[i] < r1_aligned[i]:
                # Additional confirmation: price below weekly pivot and below daily EMA50
                if close[i] < pivot_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_ok[i]:
                    signals[i] = -0.25
                    position = -1
        elif position != 0:
            # Exit conditions
            if position == 1:  # Long position
                # Exit if price breaks below S2 (stronger support) or reverses at R1
                if close[i] < s2_aligned[i] or (high[i] > r1_aligned[i] and close[i] < r1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit if price breaks above R2 (stronger resistance) or reverses at S1
                if close[i] > r2_aligned[i] or (low[i] < s1_aligned[i] and close[i] > s1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals