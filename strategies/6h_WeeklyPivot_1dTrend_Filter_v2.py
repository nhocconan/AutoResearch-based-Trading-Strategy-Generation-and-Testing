#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_1dTrend_Filter_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's H/L/C to calculate current week's pivot
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
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Volume confirmation - 60-period average volume
    vol_ma = pd.Series(volume).rolling(window=60, min_periods=60).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)  # Avoid division by zero
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or
            np.isnan(r3_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(s2_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above weekly pivot + above 1d EMA34 + volume confirmation
            if (close[i] > pivot_1w_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.2):
                # Check if we're not too far above R2 (avoid chasing extreme extension)
                if close[i] <= r2_1w_aligned[i] * 1.05:  # Within 5% above R2
                    signals[i] = 0.25
                    position = 1
            # Short conditions: price below weekly pivot + below 1d EMA34 + volume confirmation
            elif (close[i] < pivot_1w_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.2):
                # Check if we're not too far below S2 (avoid chasing extreme extension)
                if close[i] >= s2_1w_aligned[i] * 0.95:  # Within 5% below S2
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR below 1d EMA34
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR above 1d EMA34
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals