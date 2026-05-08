#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_Trend_Volume_Confirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly pivot points calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    # Resistance 1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    # Support 2 = Pivot - (High - Low)
    s2_1w = pivot_1w - (high_1w - low_1w)
    # Resistance 2 = Pivot + (High - Low)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # Support 3 = Low - 2*(High - Pivot)
    s3_1w = low_1w - 2.0 * (high_1w - pivot_1w)
    # Resistance 3 = High + 2*(Pivot - Low)
    r3_1w = high_1w + 2.0 * (pivot_1w - low_1w)
    
    # Align weekly pivots to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Volume confirmation - 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above weekly pivot + above daily EMA50 + volume confirmation
            if (close[i] > pivot_1w_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and
                vol_ratio[i] > 1.5):
                # Avoid extreme extensions beyond R2
                if close[i] <= r2_1w_aligned[i] * 1.03:  # Within 3% above R2
                    signals[i] = 0.25
                    position = 1
            # Short conditions: price below weekly pivot + below daily EMA50 + volume confirmation
            elif (close[i] < pivot_1w_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and
                  vol_ratio[i] > 1.5):
                # Avoid extreme extensions beyond S2
                if close[i] >= s2_1w_aligned[i] * 0.97:  # Within 3% below S2
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR below daily EMA50
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR above daily EMA50
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals