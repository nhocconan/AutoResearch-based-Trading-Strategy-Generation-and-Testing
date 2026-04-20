#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily pivot points and support/resistance levels
    pivot_daily = (high_daily + low_daily + close_daily) / 3
    r1_daily = 2 * pivot_daily - low_daily
    s1_daily = 2 * pivot_daily - high_daily
    r2_daily = pivot_daily + (high_daily - low_daily)
    s2_daily = pivot_daily - (high_daily - low_daily)
    r3_daily = high_daily + 2 * (pivot_daily - low_daily)
    s3_daily = low_daily - 2 * (high_daily - pivot_daily)
    r4_daily = 3 * pivot_daily + (high_daily - 3 * low_daily)
    s4_daily = 3 * pivot_daily - (3 * high_daily - low_daily)
    
    # Align daily levels to 6h timeframe
    pivot_daily_aligned = align_htf_to_ltf(prices, df_daily, pivot_daily)
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    r3_daily_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_daily_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    r4_daily_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s4_daily_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Load weekly data for regime filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Weekly pivot for regime
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_daily_aligned[i]) or np.isnan(r1_daily_aligned[i]) or 
            np.isnan(s1_daily_aligned[i]) or np.isnan(pivot_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        # Get daily levels
        pivot = pivot_daily_aligned[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r2 = r2_daily_aligned[i]
        s2 = s2_daily_aligned[i]
        r3 = r3_daily_aligned[i]
        s3 = s3_daily_aligned[i]
        r4 = r4_daily_aligned[i]
        s4 = s4_daily_aligned[i]
        pivot_weekly = pivot_weekly_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma_recent = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma_recent
        
        # Regime filter: price away from weekly pivot
        dist_from_weekly_pivot = abs(price - pivot_weekly) / pivot_weekly
        trending_regime = dist_from_weekly_pivot > 0.015  # 1.5% away from weekly pivot
        
        if position == 0:
            # Long: price breaks above R4 with volume and trending regime
            if price > r4 and vol_ok and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 with volume and trending regime
            elif price < s4 and vol_ok and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below R1 or regime turns ranging
            if price < r1 or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above S1 or regime turns ranging
            if price > s1 or not trending_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_Pivot_R4S4_Breakout_VolumeTrendFilter_v1"
timeframe = "6h"
leverage = 1.0