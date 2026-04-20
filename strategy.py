#!/usr/bin/env python3
"""
1d_Pivot_R1S1_Breakout_Volume
Hypothesis: On 1d timeframe, buy when price breaks above weekly pivot R1 with volume confirmation,
sell when price breaks below weekly pivot S1. Uses weekly pivot levels for structure and
volume surge to confirm breakout strength. Works in bull markets (breakouts continue) and
bear markets (breakdowns continue) by following institutional interest at key levels.
Target: 15-25 trades/year with position size 0.25 to manage drawdown.
"""

name = "1d_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points
    _, weekly_r1, _, _, weekly_s1, _, _ = calculate_pivot_points(
        weekly_high, weekly_low, weekly_close
    )
    
    # Align weekly pivot levels to daily timeframe (wait for weekly close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Calculate volume average (20-period) for spike detection
    vol_ma = np.zeros_like(volume)
    vol_ma[:] = np.nan
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 00-24 UTC (full day for 1d timeframe)
    # For 1d, we use all hours as each bar represents a full day
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume spike
            if close[i] > weekly_r1_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume spike
            elif close[i] < weekly_s1_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 (failed breakout) OR
            # price returns to pivot level (take profit)
            if close[i] < weekly_s1_aligned[i] or close[i] < (weekly_r1_aligned[i] + weekly_s1_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 (failed breakdown) OR
            # price returns to pivot level (take profit)
            if close[i] > weekly_r1_aligned[i] or close[i] > (weekly_r1_aligned[i] + weekly_s1_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals