#!/usr/bin/env python3
name = "1d_WeeklyPivot_BullBear_Mode"
timeframe = "1d"
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
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot points (using previous week's OHLC)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_open = df_1w['open'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Trend filter: 21-day EMA (using daily data)
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21)  # Ensure enough data for EMA and pivots
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema21[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot + above EMA21 + volume confirmation
            if close[i] > pivot_aligned[i] and close[i] > ema21[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot + below EMA21 + volume confirmation
            elif close[i] < pivot_aligned[i] and close[i] < ema21[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below weekly support OR below EMA21
            if close[i] < s1_aligned[i] or close[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above weekly resistance OR above EMA21
            if close[i] > r1_aligned[i] or close[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals