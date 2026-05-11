#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_Trend_Volume"
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
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pivot_w = (high_w + low_w + close_w) / 3
    r1 = 2 * pivot_w - low_w
    s1 = 2 * pivot_w - high_w
    r2 = pivot_w + (high_w - low_w)
    s2 = pivot_w - (high_w - low_w)
    r3 = high_w + 2 * (pivot_w - low_w)
    s3 = low_w - 2 * (high_w - pivot_w)
    
    # Weekly EMA50 for trend filter
    ema_50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_w)
    
    # Volume filter: 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(ema_50_w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false breakouts
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Price breaks above R2 with volume and above weekly EMA50
            if (close[i] > r2_aligned[i] and 
                volume_surge and 
                close[i] > ema_50_w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S2 with volume and below weekly EMA50
            elif (close[i] < s2_aligned[i] and 
                  volume_surge and 
                  close[i] < ema_50_w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite pivot level
            if position == 1:
                # Exit long: price touches or goes below S1
                if close[i] <= s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches or goes above R1
                if close[i] >= r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals