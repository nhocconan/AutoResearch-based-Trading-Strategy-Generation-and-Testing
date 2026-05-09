#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d1w_PivotZone_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d data for pivot calculation (using previous day)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # 1w data for pivot calculation (using previous week)
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    # Calculate 1d pivots (standard)
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    r2_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    r3_1d = prev_high_1d + 2 * (pivot_1d - prev_low_1d)
    s3_1d = prev_low_1d - 2 * (prev_high_1d - pivot_1d)
    
    # Calculate 1w pivots
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1_1w = 2 * pivot_1w - prev_low_1w
    s1_1w = 2 * pivot_1w - prev_high_1w
    
    # Trend: 1w EMA21
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: current 1d volume > 1.3 * 20-day average
    vol_series_1d = pd.Series(df_1d['volume'].values)
    vol_ma_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma_1d * 1.3)
    
    # Align all to 6h
    pivot_1d_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1w_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema21_1w_6h = align_htf_to_ltf(prices, df_1w, ema21_1w)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_6h[i]) or np.isnan(r1_1d_6h[i]) or np.isnan(s1_1d_6h[i]) or
            np.isnan(r2_1d_6h[i]) or np.isnan(s2_1d_6h[i]) or np.isnan(r3_1d_6h[i]) or
            np.isnan(s3_1d_6h[i]) or np.isnan(pivot_1w_6h[i]) or np.isnan(r1_1w_6h[i]) or
            np.isnan(s1_1w_6h[i]) or np.isnan(ema21_1w_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_1d_val = pivot_1d_6h[i]
        r1_1d_val = r1_1d_6h[i]
        s1_1d_val = s1_1d_6h[i]
        r2_1d_val = r2_1d_6h[i]
        s2_1d_val = s2_1d_6h[i]
        r3_1d_val = r3_1d_6h[i]
        s3_1d_val = s3_1d_6h[i]
        pivot_1w_val = pivot_1w_6h[i]
        r1_1w_val = r1_1w_6h[i]
        s1_1w_val = s1_1w_6h[i]
        trend = ema21_1w_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: price above 1w pivot and 1d R2, with volume and above weekly trend
            if (close[i] > pivot_1w_val and close[i] > r2_1d_val and 
                close[i] > trend and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: price below 1w pivot and 1d S2, with volume and below weekly trend
            elif (close[i] < pivot_1w_val and close[i] < s2_1d_val and 
                  close[i] < trend and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1d S1 or below 1w pivot
            if close[i] < s1_1d_val or close[i] < pivot_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1d R1 or above 1w pivot
            if close[i] > r1_1d_val or close[i] > pivot_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals