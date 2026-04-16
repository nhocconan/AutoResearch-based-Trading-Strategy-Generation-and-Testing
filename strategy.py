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
    
    # === 1d data (primary) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === Weekly data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === Daily pivot points (standard calculation) ===
    pivot_1d = np.zeros_like(close_1d)
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        pp = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        pivot_1d[i] = pp
        r1_1d[i] = 2 * pp - low_1d[i-1]
        s1_1d[i] = 2 * pp - high_1d[i-1]
    
    # === Weekly trend filter: price vs weekly pivot ===
    weekly_pivot = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        pp = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        weekly_pivot[i] = pp
    
    weekly_trend = np.where(close_1w > weekly_pivot, 1, -1)
    
    # === Daily volume ratio for confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # Align all HTF data to 1d timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for pivots and volume MA
    warmup = 25
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pt = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        wt = weekly_trend_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
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
            # LONG: Price above R1 with volume confirmation and weekly uptrend
            if price > r1 and wt == 1 and vol_ratio > 2.0:
                signals[i] = 0.25
                position = 1
                continue
            # SHORT: Price below S1 with volume confirmation and weekly downtrend
            elif price < s1 and wt == -1 and vol_ratio > 2.0:
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

name = "1d_Pivot_R1S1_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0