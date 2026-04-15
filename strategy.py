#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Weekly Pivot Breakout with Volume and Trend Filter
# Uses weekly pivot levels from 1w data as support/resistance. 
# Long when price breaks above weekly pivot resistance with volume spike and bullish trend (12h close > 20 EMA).
# Short when price breaks below weekly pivot support with volume spike and bearish trend (12h close < 20 EMA).
# Weekly timeframe reduces noise, pivot levels provide institutional reference points.
# Works in bull markets (breaks up) and bear markets (breaks down). Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # Resistance 1 = (2 * Pivot) - L
    # Support 1 = (2 * Pivot) - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = (2 * pivot_1w) - low_1w
    s1_1w = (2 * pivot_1w) - high_1w
    
    # Previous week's levels (shifted by 1 to avoid look-ahead)
    pivot_prev = np.roll(pivot_1w, 1)
    r1_prev = np.roll(r1_1w, 1)
    s1_prev = np.roll(s1_1w, 1)
    pivot_prev[0] = np.nan
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Align weekly pivot levels to 12h timeframe
    pivot_prev_aligned = align_htf_to_ltf(prices, df_1w, pivot_prev)
    r1_prev_aligned = align_htf_to_ltf(prices, df_1w, r1_prev)
    s1_prev_aligned = align_htf_to_ltf(prices, df_1w, s1_prev)
    
    # Load 12h data for trend filter (20-period EMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 20-period EMA on 12h
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align EMA to 12h timeframe (no additional delay needed for EMA)
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_prev_aligned[i]) or np.isnan(r1_prev_aligned[i]) or 
            np.isnan(s1_prev_aligned[i]) or np.isnan(ema_20_aligned[i])):
            continue
        
        # Long entry: price breaks above weekly R1 + volume spike + bullish trend (close > EMA20)
        if (close[i] > r1_prev_aligned[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-10):i+1]) and
            close[i] > ema_20_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below weekly S1 + volume spike + bearish trend (close < EMA20)
        elif (close[i] < s1_prev_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-10):i+1]) and
              close[i] < ema_20_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse breakout or trend change
        elif position == 1 and (close[i] < pivot_prev_aligned[i] or close[i] < ema_20_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > pivot_prev_aligned[i] or close[i] > ema_20_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WeeklyPivot_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0