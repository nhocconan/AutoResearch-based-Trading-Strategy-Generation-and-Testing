#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    # Calculate weekly pivot points
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Resistance levels
    r1_w = 2 * pivot_w - low_w
    r2_w = pivot_w + (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    # Support levels
    s1_w = 2 * pivot_w - high_w
    s2_w = pivot_w - (high_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Align weekly pivot levels to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    
    # Weekly trend filter: price above/below pivot
    trend_w = close_w > pivot_w  # True for uptrend, False for downtrend
    trend_w_aligned = align_htf_to_ltf(prices, df_1w, trend_w.astype(float))
    
    # Daily volume filter (to avoid low-volume false breakouts)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or 
            np.isnan(s3_w_aligned[i]) or np.isnan(trend_w_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio_1d_aligned[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R3 with volume and weekly uptrend
            if (close[i] > r3_w_aligned[i] and 
                volume_surge and 
                trend_w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume and weekly downtrend
            elif (close[i] < s3_w_aligned[i] and 
                  volume_surge and 
                  trend_w_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to weekly pivot level
            if position == 1:
                # Exit long: price touches or goes below weekly pivot
                if close[i] <= pivot_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes above weekly pivot
                if close[i] >= pivot_w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals