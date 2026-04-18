#!/usr/bin/env python3
"""
6h Weekly Pivot Reversal with Volume Confirmation
Strategy: Buy near weekly S1/S2 support with volume spike, sell near weekly R1/R2 resistance with volume spike.
          Use daily EMA50 as trend filter to avoid counter-trend trades.
          Designed for mean reversion in ranging markets and breakout continuation in trending markets.
"""

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
    
    # Get weekly data for pivot points (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Align daily EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long near S1/S2 with volume spike and above daily EMA50 (bullish bias)
            if (price <= s1_level * 1.005 or price <= s2_level * 1.005) and volume_spike[i] and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short near R1/R2 with volume spike and below daily EMA50 (bearish bias)
            elif (price >= r1_level * 0.995 or price >= r2_level * 0.995) and volume_spike[i] and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price reaches pivot or R1, or breaks below daily EMA50
            if price >= pivot_level * 0.995 or price >= r1_level * 0.995 or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price reaches pivot or S1, or breaks above daily EMA50
            if price <= pivot_level * 1.005 or price <= s1_level * 1.005 or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Reversal_Volume_EMA50"
timeframe = "6h"
leverage = 1.0