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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === Daily data for pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 6h Donchian channel (20-period) ===
    donchian_high = np.full_like(close_6h, np.nan)
    donchian_low = np.full_like(close_6h, np.nan)
    for i in range(20, len(close_6h)):
        donchian_high[i] = np.max(high_6h[i-19:i+1])
        donchian_low[i] = np.min(low_6h[i-19:i+1])
    
    # === Daily pivot points (standard calculation) ===
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
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
    
    # === Weekly trend: price vs weekly pivot ===
    weekly_pivot = np.full_like(close_1w, np.nan)
    for i in range(1, len(close_1w)):
        pp = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        weekly_pivot[i] = pp
    
    weekly_trend = np.where(close_1w > weekly_pivot, 1, -1)
    
    # === 6h volume ratio (20-period) for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    # Align all HTF data to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
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
    vol_ratio_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio_6h)
    
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
            np.isnan(vol_ratio_6h_aligned[i])):
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
        vol_ratio = vol_ratio_6h_aligned[i]
        
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

name = "6h_Donchian_Pivot_R1S1_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0