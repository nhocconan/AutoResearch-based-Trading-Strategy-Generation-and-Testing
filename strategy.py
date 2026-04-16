#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for trend and pivot) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Weekly data for pivot direction ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 4h Donchian channel (20-period) ===
    donchian_high = np.zeros_like(close_4h)
    donchian_low = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if i < 20:
            donchian_high[i] = np.max(high_4h[:i+1])
            donchian_low[i] = np.min(low_4h[:i+1])
        else:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # === 1d daily pivot points (standard calculation) ===
    # Using previous day's OHLC for today's pivot
    pivot_1d = np.zeros_like(close_1d)
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    r2_1d = np.zeros_like(close_1d)
    s2_1d = np.zeros_like(close_1d)
    r3_1d = np.zeros_like(close_1d)
    s3_1d = np.zeros_like(close_1d)
    r4_1d = np.zeros_like(close_1d)
    s4_1d = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        pivot_1d[i] = pp
        r1_1d[i] = 2 * pp - low_1d[i-1]
        s1_1d[i] = 2 * pp - high_1d[i-1]
        r2_1d[i] = pp + (high_1d[i-1] - low_1d[i-1])
        s2_1d[i] = pp - (high_1d[i-1] - low_1d[i-1])
        r3_1d[i] = high_1d[i-1] + 2 * (pp - low_1d[i-1])
        s3_1d[i] = low_1d[i-1] - 2 * (high_1d[i-1] - pp)
        r4_1d[i] = r3_1d[i] + (high_1d[i-1] - low_1d[i-1])
        s4_1d[i] = s3_1d[i] - (high_1d[i-1] - low_1d[i-1])
    
    # === Weekly trend filter: price vs weekly pivot ===
    weekly_pivot = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        pp = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        weekly_pivot[i] = pp
    
    weekly_trend = np.where(close_1w > weekly_pivot, 1, -1)
    
    # === 4h volume ratio for confirmation ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_4h = volume_4h / vol_ma_20_4h
    
    # Align all HTF data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    vol_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ratio_4h)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and pivots
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r2_1d_aligned[i]) or 
            np.isnan(s2_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(vol_ratio_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        pt = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        wt = weekly_trend_aligned[i]
        vol_ratio = vol_ratio_4h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below daily pivot OR weekly trend turns bearish
            if price < pt or wt == -1:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above daily pivot OR weekly trend turns bullish
            if price > pt or wt == 1:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Donchian breakout above R1 with volume and weekly uptrend
            if price > r1 and price > dh and wt == 1 and vol_ratio > 1.8:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Donchian breakout below S1 with volume and weekly downtrend
            elif price < s1 and price < dl and wt == -1 and vol_ratio > 1.8:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Pivot_R1S1_WeeklyTrend_Volume"
timeframe = "4h"
leverage = 1.0