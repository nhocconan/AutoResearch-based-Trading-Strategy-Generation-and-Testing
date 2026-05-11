#!/usr/bin/env python3
name = "6h_WeeklyPivot_Trend_DoubleConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot point (P) and support/resistance levels
    # P = (H + L + C) / 3
    # S1 = 2*P - H
    # R1 = 2*P - L
    # S2 = P - (H - L)
    # R2 = P + (H - L)
    # S3 = L - 2*(H - P)
    # R3 = H + 2*(P - L)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - high_1w
    s1 = 2 * pivot - low_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly pivot levels to 6h timeframe (using previous week's values)
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot AND above 1d EMA50 (uptrend) AND volume surge
            # Enter near R1/S1 area with confirmation
            if close[i] > pivot_6h[i] and close[i] > ema_1d_aligned[i] and volume_filter[i]:
                # Additional confirmation: price should be in upper half of weekly range
                weekly_range = r1_6h[i] - s1_6h[i]
                if weekly_range > 0 and close[i] > (s1_6h[i] + weekly_range * 0.5):
                    signals[i] = 0.25
                    position = 1
            # Short: price below weekly pivot AND below 1d EMA50 (downtrend) AND volume surge
            elif close[i] < pivot_6h[i] and close[i] < ema_1d_aligned[i] and volume_filter[i]:
                # Additional confirmation: price should be in lower half of weekly range
                weekly_range = r1_6h[i] - s1_6h[i]
                if weekly_range > 0 and close[i] < (r1_6h[i] - weekly_range * 0.5):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price falls below weekly pivot OR below 1d EMA50 (trend change)
            if close[i] < pivot_6h[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above weekly pivot OR above 1d EMA50 (trend change)
            if close[i] > pivot_6h[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals