#!/usr/bin/env python3
"""
12h_WPivot_R1S1_Breakout_Volume_Filter
Strategy: 12h Camarilla pivot R1/S1 breakout + volume confirmation (1.5x 20-bar avg) + 1d Chop regime filter (>61.8 = range).
Long: Price breaks above R1 + volume filter + 1d Chop > 61.8
Short: Price breaks below S1 + volume filter + 1d Chop > 61.8
Exit: Price crosses back through pivot point (mean reversion in chop)
Position size: 0.25
Uses 12h for structure, volume for confirmation, 1d Chop for regime filter to avoid trending chop failures.
Designed to work in both bull and bear markets by filtering for range-bound conditions.
"""

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
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1, pivot) on 12h
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12
    s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12
    
    # Align 12h pivots to main timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Get 12h volume for confirmation
    volume_12h = df_12h['volume'].values
    volume_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma20_12h)
    
    # Get 1d data for Chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chop(14) on 1d: 100 * log10(sum(ATR(1)) / (max(high)-min(low))) / log10(14)
    atr_1 = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1[0] = high_1d[0] - low_1d[0]  # first value
    sum_atr = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    roll_max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    roll_min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = roll_max_high - roll_min_low
    chop_1d = 100 * (np.log10(sum_atr) - np.log10(range_14)) / np.log10(14)
    chop_1d = np.where(range_14 > 0, chop_1d, 50)  # avoid division by zero
    
    # Align 1d Chop to main timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):  # warmup for 20-bar volume MA
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_ma20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume aligned to main timeframe
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        volume_filter = vol_12h_current > (1.5 * volume_ma20_12h_aligned[i])
        chop_filter = chop_1d_aligned[i] > 61.8  # range regime on 1d
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        # Exit conditions: price crosses back through pivot point
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 + volume + chop (range)
            if breakout_up and volume_filter and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume + chop (range)
            elif breakout_down and volume_filter and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below pivot
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above pivot
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WPivot_R1S1_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0