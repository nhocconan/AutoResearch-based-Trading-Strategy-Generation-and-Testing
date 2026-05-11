# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly pivot structure and 1d trend filter.
- Uses weekly pivot points (calculated from prior week OHLC) as key support/resistance
- Long when price breaks above weekly R1 with 1d uptrend (EMA50) and volume confirmation
- Short when price breaks below weekly S1 with 1d downtrend and volume confirmation
- Weekly pivot provides structural levels that work in both trending and ranging markets
- Volume confirmation filters false breakouts
- Target: 20-50 trades/year (~80-200 total over 4 years) to minimize fee drag
"""

name = "6h_WeeklyPivot_R1S1_Breakout_1dTrend_Volume"
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot, R1, S1 from prior week
    pivot = np.zeros(len(high_1w))
    R1 = np.zeros(len(high_1w))
    S1 = np.zeros(len(high_1w))
    
    for i in range(len(high_1w)):
        if i < 1:
            pivot[i] = np.nan
            R1[i] = np.nan
            S1[i] = np.nan
        else:
            # Prior week's OHLC
            prev_high = high_1w[i-1]
            prev_low = low_1w[i-1]
            prev_close = close_1w[i-1]
            
            # Standard pivot point calculation
            pivot[i] = (prev_high + prev_low + prev_close) / 3.0
            R1[i] = 2 * pivot[i] - prev_low
            S1[i] = 2 * pivot[i] - prev_high
    
    # Get daily trend filter (1d EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # Align weekly pivots and daily trend to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 + 1d uptrend + volume confirmation
            if (close[i] > R1_aligned[i] and 
                trend_up_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 + 1d downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 or trend changes to down
            if (close[i] < S1_aligned[i] or not trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 or trend changes to up
            if (close[i] > R1_aligned[i] or trend_up_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals