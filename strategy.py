#!/usr/bin/env python3
"""
6h_WeeklyPivot_1d_Trend_SR_Bounce
Hypothesis: Bounce off weekly pivot support/resistance levels in the direction of 1d EMA trend.
Uses weekly pivot levels (calculated from prior week) as dynamic S/R, with trend filter from 1d EMA50.
Targets 12-37 trades per year by requiring confluence of weekly S/R bounce and daily trend alignment.
Works in bull markets by buying dips to weekly support in uptrend, and in bear markets by selling rallies to weekly resistance in downtrend.
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
    
    # Get weekly data for pivot levels (HTF)
    df_week = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    high_week = df_week['high'].values
    low_week = df_week['low'].values
    close_week = df_week['close'].values
    
    pivot_week = (high_week + low_week + close_week) / 3
    r1_week = (2 * pivot_week) - high_week
    s1_week = (2 * pivot_week) - low_week
    
    # Align weekly levels to 6h timeframe (wait for bar close)
    pivot_week_aligned = align_htf_to_ltf(prices, df_week, pivot_week)
    r1_week_aligned = align_htf_to_ltf(prices, df_week, r1_week)
    s1_week_aligned = align_htf_to_ltf(prices, df_week, s1_week)
    
    # Get 1d trend (EMA50) for directional bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_week_aligned[i]) or np.isnan(r1_week_aligned[i]) or 
            np.isnan(s1_week_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price bounces above weekly S1 in 1d uptrend
            if (close[i] > s1_week_aligned[i] and 
                close[i] < pivot_week_aligned[i] and  # inside S1-Pivot range
                close[i] > ema_1d_aligned[i]):       # above 1d EMA50 (uptrend)
                signals[i] = 0.25
                position = 1
            # Short entry: price bounces below weekly R1 in 1d downtrend
            elif (close[i] < r1_week_aligned[i] and 
                  close[i] > pivot_week_aligned[i] and  # inside Pivot-R1 range
                  close[i] < ema_1d_aligned[i]):       # below 1d EMA50 (downtrend)
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches weekly pivot or 1d trend turns down
            if (close[i] >= pivot_week_aligned[i] or 
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly pivot or 1d trend turns up
            if (close[i] <= pivot_week_aligned[i] or 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_1d_Trend_SR_Bounce"
timeframe = "6h"
leverage = 1.0