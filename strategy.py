#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_DailyTrend_Volume"
timeframe = "6h"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot (Classic): P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h
    pivot_align = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_align = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_align = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_align = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_align = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_align = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_align = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_align = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_align[i]) or np.isnan(r1_align[i]) or np.isnan(s1_align[i]) or
            np.isnan(r2_align[i]) or np.isnan(s2_align[i]) or np.isnan(r3_align[i]) or
            np.isnan(s3_align[i]) or np.isnan(ema50_1d_align[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R2, uptrend (price > EMA50), volume spike
            if (close[i] > r2_align[i] and 
                close[i] > ema50_1d_align[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2, downtrend (price < EMA50), volume spike
            elif (close[i] < s2_align[i] and 
                  close[i] < ema50_1d_align[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S2 (opposite level)
            if close[i] < s2_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R2 (opposite level)
            if close[i] > r2_align[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals