#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-day pivot-based breakout and weekly trend filter.
In bull markets, price breaks above weekly pivot resistance with volume confirmation.
In bear markets, price breaks below weekly pivot support with volume confirmation.
Uses weekly pivot levels from prior week to avoid look-ahead, combined with 6 Donchian breakout
for entry confirmation. Volume spike filter reduces false breaks.
Designed for low trade frequency (12-37/year) to minimize fee drag.
"""

name = "6h_WeeklyPivot_Breakout_1dTrend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly data for pivot levels (using 1w as HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week (HLC of previous week)
    # Use rolling window to get previous week's HLC
    weekly_high = df_1w['high'].rolling(window=2, min_periods=2).max().shift(1).values  # Previous week high
    weekly_low = df_1w['low'].rolling(window=2, min_periods=2).min().shift(1).values    # Previous week low
    weekly_close = df_1w['close'].rolling(window=2, min_periods=2).mean().shift(1).values  # Previous week close
    
    # Standard pivot point calculation: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian breakout (6-period for 6f timeframe)
    high_roll = pd.Series(high).rolling(window=6, min_periods=6).max().values
    low_roll = pd.Series(low).rolling(window=6, min_periods=6).min().values
    
    # Volume spike (24-period average = 4 days of 6h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34  # Ensure EMA34 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above weekly R1, above 1d EMA34, volume spike
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1, below 1d EMA34, volume spike
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below weekly S1 or below 1d EMA34
            if close[i] < weekly_s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above weekly R1 or above 1d EMA34
            if close[i] > weekly_r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals