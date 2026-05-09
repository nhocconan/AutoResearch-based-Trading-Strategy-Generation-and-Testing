#!/usr/bin/env python3
name = "6H_WeeklyPivot_VolumeTrend"
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
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = np.full_like(close_1w, np.nan)
    r1 = np.full_like(close_1w, np.nan)
    s1 = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        weekly_high = high_1w[i]
        weekly_low = low_1w[i]
        weekly_close = close_1w[i]
        pivot[i] = (weekly_high + weekly_low + weekly_close) / 3.0
        r1[i] = 2 * pivot[i] - weekly_low
        s1[i] = 2 * pivot[i] - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume EMA20
    vol_ema20_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h volume EMA20 to 6h timeframe
    vol_ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ema20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        # Downtrend: price below 1d EMA50
        downtrend = close[i] < ema50_1d_aligned[i]
        # Volume surge: current volume > 2.0x 12h volume EMA20
        volume_surge = volume[i] > vol_ema20_12h_aligned[i] * 2.0
        
        if position == 0:
            # Enter long: Uptrend + price breaks above weekly R1 + volume surge
            if uptrend and close[i] > r1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below weekly S1 + volume surge
            elif downtrend and close[i] < s1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below weekly pivot
            if not uptrend or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above weekly pivot
            if not downtrend or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals