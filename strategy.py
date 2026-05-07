#!/usr/bin/env python3
name = "6h_WeeklyPivot_Direction_1dVolatility_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Shift to get previous week's values
    high_w_prev = np.roll(high_w, 1)
    low_w_prev = np.roll(low_w, 1)
    close_w_prev = np.roll(close_w, 1)
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_w = (high_w_prev + low_w_prev + close_w_prev) / 3
    # Weekly support/resistance levels
    r1_w = 2 * pivot_w - low_w_prev
    s1_w = 2 * pivot_w - high_w_prev
    r2_w = pivot_w + (high_w_prev - low_w_prev)
    s2_w = pivot_w - (high_w_prev - low_w_prev)
    
    # Align weekly levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Get daily data for volatility filter (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - Average True Range
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily ATR ratio (current ATR vs 50-period average) for volatility regime
    atr_ma50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ma50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma50)
    atr_ratio = atr_14_aligned / atr_ma50_aligned  # >1 = above average volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in high volatility regimes (ATR ratio > 1.2)
        if atr_ratio[i] <= 1.2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R2 with bullish bias (above weekly pivot)
            if (close[i] > r2_w_aligned[i] and 
                close[i] > pivot_w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S2 with bearish bias (below weekly pivot)
            elif (close[i] < s2_w_aligned[i] and 
                  close[i] < pivot_w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 or volatility drops
            if (close[i] < s1_w_aligned[i] or 
                atr_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 or volatility drops
            if (close[i] > r1_w_aligned[i] or 
                atr_ratio[i] < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals