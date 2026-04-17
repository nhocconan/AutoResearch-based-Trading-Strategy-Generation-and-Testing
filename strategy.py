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
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + range_1d
    s2_1d = pivot_1d - range_1d
    
    # Align 1d pivot levels to 4h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Get 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Choppiness filter: avoid trading in high chop (range-bound) markets
    # Calculate Choppiness Index (14-period)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.append([np.nan], close[:-1]))), np.abs(low - np.append([np.nan], close[:-1])))).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10((atr_14 * 14) / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_filter = chop < 61.8  # Only trade when NOT in chop (trending market)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume, above 1d EMA50, and NOT in chop
            if close[i] > r1_1d_aligned[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below 1d EMA50, and NOT in chop
            elif close[i] < s1_1d_aligned[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below pivot OR below S1
            if close[i] < pivot_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above pivot OR above R1
            if close[i] > pivot_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dPivot_R1S1_Volume_EMA50_ChopFilter"
timeframe = "4h"
leverage = 1.0