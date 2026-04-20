#!/usr/bin/env python3
# 4h_1d_WeeklyPivot_R1S1_Breakout_VolumeTrend
# Hypothesis: Use weekly pivot levels (R1/S1) from 1W data with 4h breakout, volume confirmation, and 1d EMA50 trend filter.
# Weekly pivots provide stronger support/resistance than daily, reducing false breakouts. Trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional interest. Designed for fewer trades (~20-40/year) to minimize fee drag in both bull and bear markets.

name = "4h_1d_WeeklyPivot_R1S1_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly typical price for pivot
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly pivot and R1/S1 levels
    pivot_1w = typical_price_1w
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    
    # Align weekly levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume spike and uptrend (close > daily EMA50)
            if (close[i] > r1_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume spike and downtrend (close < daily EMA50)
            elif (close[i] < s1_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend reverses (close < daily EMA50)
            if close[i] < s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend reverses (close > daily EMA50)
            if close[i] > r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals