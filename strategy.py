#!/usr/bin/env python3
"""
6h_WeeklyPivot_R4S4_Breakout_VolumeFilter
Hypothesis: Weekly pivot levels (R4/S4) act as strong support/resistance. Breakouts above R4 or below S4 with volume confirmation indicate strong momentum. Weekly trend filter ensures trades align with higher timeframe direction. This captures breakout moves while avoiding false breakouts in ranging markets. Works in both bull and bear markets by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate weekly pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    r4 = r3 + (high - low)
    s4 = s3 - (high - low)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate volume moving average for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    weekly_pivot = np.full(len(weekly_high), np.nan)
    weekly_r4 = np.full(len(weekly_high), np.nan)
    weekly_s4 = np.full(len(weekly_high), np.nan)
    
    for i in range(len(weekly_high)):
        _, _, _, _, r4, _, _, _, s4 = calculate_pivot_points(
            weekly_high[i], weekly_low[i], weekly_close[i]
        )
        weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
        weekly_r4[i] = r4
        weekly_s4[i] = s4
    
    # Align weekly data to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Weekly trend: price above/below pivot
    weekly_trend_up = np.full(len(pivot_aligned), False)
    weekly_trend_down = np.full(len(pivot_aligned), False)
    for i in range(len(pivot_aligned)):
        if not np.isnan(pivot_aligned[i]):
            weekly_trend_up[i] = close[i] > pivot_aligned[i]
            weekly_trend_down[i] = close[i] < pivot_aligned[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: break above R4 with volume, weekly trend up
            if (close[i] > r4_aligned[i] and volume_confirmed and 
                weekly_trend_up[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: break below S4 with volume, weekly trend down
            elif (close[i] < s4_aligned[i] and volume_confirmed and 
                  weekly_trend_down[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price falls back below pivot or weekly trend changes
            if (close[i] < pivot_aligned[i] or not weekly_trend_up[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above pivot or weekly trend changes
            if (close[i] > pivot_aligned[i] or not weekly_trend_down[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R4S4_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0