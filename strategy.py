#!/usr/bin/env python3
"""
6h_WeeklyPivot_DailyTrend_Filter
Hypothesis: Use weekly pivot points for long-term bias and daily EMA trend filter for entry timing.
Long when price is above weekly pivot AND above daily EMA50, short when below weekly pivot AND below daily EMA50.
Weekly pivot provides structural support/resistance from higher timeframe, daily EMA filters for momentum alignment.
Designed for 6h timeframe to capture medium-term swings with low frequency. Works in bull markets by buying
above pivot/EMA alignment and in bear markets by selling below pivot/EMA alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Load daily data for EMA50 trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    # Calculate EMA50 on daily timeframe
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: price above weekly pivot AND above daily EMA50
            if price > weekly_pivot_aligned[i] and price > ema50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: price below weekly pivot AND below daily EMA50
            elif price < weekly_pivot_aligned[i] and price < ema50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR below daily EMA50
            if price < weekly_pivot_aligned[i] or price < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR above daily EMA50
            if price > weekly_pivot_aligned[i] or price > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DailyTrend_Filter"
timeframe = "6h"
leverage = 1.0