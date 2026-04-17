#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter
Strategy: 6h weekly pivot point (R1/S1) breakout with volume filter.
Long: Price breaks above weekly pivot R1 + volume > 1.5x 20-period avg
Short: Price breaks below weekly pivot S1 + volume > 1.5x 20-period avg
Exit: Opposite pivot level break
Position size: 0.25
Uses weekly pivot levels for structure, volume for confirmation.
Designed to work in both bull and bear markets by capturing breakouts from key levels.
Timeframe: 6h
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
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Calculate 6h volume average (20-period)
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    volume_ma20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_ma20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma20_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h volume aligned to 6h
        vol_6h_current = align_htf_to_ltf(prices, df_6h, volume_6h)[i]
        volume_filter = vol_6h_current > (1.5 * volume_ma20_6h_aligned[i])
        
        # Breakout signals
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: Breakout above R1 + volume filter
            if breakout_up and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 + volume filter
            elif breakout_down and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Breakdown below S1 (opposite level)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Breakout above R1 (opposite level)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0