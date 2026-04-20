#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_Trend_Filter
Hypothesis: Trade breakouts from weekly pivot levels (R1/S1) on 6h timeframe with daily trend filter.
Weekly pivot levels act as institutional support/resistance. Breakouts above R1 with daily uptrend or
below S1 with daily downtrend capture institutional flow. Uses volume confirmation to avoid false breakouts.
Designed for low frequency (target 20-50 trades/year) to minimize fee impact in both bull and bear markets.
"""

name = "6h_WeeklyPivot_R1_S1_Breakout_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's data)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # R1 = 2*P - L
    r1_w = 2 * pivot_w - low_w
    # S1 = 2*P - H
    s1_w = 2 * pivot_w - high_w
    
    # Align weekly levels to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_w, s1_w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_confirm = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume confirmation and daily uptrend
            if close[i] > r1_w_aligned[i] and volume_confirm[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume confirmation and daily downtrend
            elif close[i] < s1_w_aligned[i] and volume_confirm[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 OR daily trend turns down
            if close[i] < s1_w_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 OR daily trend turns up
            if close[i] > r1_w_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals