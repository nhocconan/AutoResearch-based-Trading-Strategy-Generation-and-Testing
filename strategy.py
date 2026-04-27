#!/usr/bin/env python3
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
    
    # Get 12h data for Donchian channels and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate Donchian channel (20-period) on 12h
    upper_20_12h = np.full(len(df_12h), np.nan)
    lower_20_12h = np.full(len(df_12h), np.nan)
    
    for i in range(20, len(df_12h)):
        upper_20_12h[i] = np.max(high_12h[i-20:i])
        lower_20_12h[i] = np.min(low_12h[i-20:i])
    
    # Align Donchian levels to 6h timeframe
    upper_20_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_20_12h)
    lower_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_20_12h)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    for i in range(20, len(df_12h)):
        vol_ma_20_12h[i] = np.mean(vol_12h[i-20:i])
    
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Get daily data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using prior week's data
    # We'll use the prior week's high, low, close to calculate pivots
    weekly_high = np.full(len(df_1d), np.nan)
    weekly_low = np.full(len(df_1d), np.nan)
    weekly_close = np.full(len(df_1d), np.nan)
    
    # Calculate weekly aggregates (simplified - using 5-day week)
    for i in range(5, len(df_1d)):
        weekly_high[i] = np.max(high_1d[i-5:i])
        weekly_low[i] = np.min(low_1d[i-5:i])
        weekly_close[i] = close_1d[i-1]  # Previous day's close as weekly close
    
    # Calculate pivot points and support/resistance levels
    pivot = np.full(len(df_1d), np.nan)
    r1 = np.full(len(df_1d), np.nan)
    s1 = np.full(len(df_1d), np.nan)
    r2 = np.full(len(df_1d), np.nan)
    s2 = np.full(len(df_1d), np.nan)
    r3 = np.full(len(df_1d), np.nan)
    s3 = np.full(len(df_1d), np.nan)
    r4 = np.full(len(df_1d), np.nan)
    s4 = np.full(len(df_1d), np.nan)
    
    for i in range(5, len(df_1d)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            r1[i] = 2 * pivot[i] - weekly_low[i]
            s1[i] = 2 * pivot[i] - weekly_high[i]
            r2[i] = pivot[i] + (weekly_high[i] - weekly_low[i])
            s2[i] = pivot[i] - (weekly_high[i] - weekly_low[i])
            r3[i] = weekly_high[i] + 2 * (pivot[i] - weekly_low[i])
            s3[i] = weekly_low[i] - 2 * (weekly_high[i] - pivot[i])
            r4[i] = r3[i] + (weekly_high[i] - weekly_low[i])
            s4[i] = s3[i] - (weekly_high[i] - weekly_low[i])
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup period
    start_idx = max(20, 5)
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_12h_aligned[i]) or 
            np.isnan(lower_20_12h_aligned[i]) or
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 12h volume MA
        volume_confirm = vol > 1.5 * vol_ma if vol_ma > 0 else False
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume confirmation
            if (price > r4_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with volume confirmation
            elif (price < s4_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below R1 or volume drops significantly
            if (price < r1_aligned[i] or vol < 0.5 * vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S1 or volume drops significantly
            if (price > s1_aligned[i] or vol < 0.5 * vol_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DonchianBreakout_Volume_WeeklyPivotR4S4_v1"
timeframe = "6h"
leverage = 1.0