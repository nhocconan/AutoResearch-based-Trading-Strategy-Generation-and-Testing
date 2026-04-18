#!/usr/bin/env python3
"""
6h_WeeklyPivot_Direction_1dVolumeFilter
Hypothesis: Use weekly pivot points (from previous week) to determine trend direction, and daily volume filter to confirm strength. 
Go long when price is above weekly pivot AND daily volume > 1.5x 20-day average, short when price is below weekly pivot AND daily volume > 1.5x 20-day average.
Weekly pivots provide structural support/resistance that works in both bull and bear markets, while volume filter ensures we only trade during active participation.
Target: 15-30 trades/year by using weekly pivots (low frequency) and volume confirmation to reduce noise.
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
    
    # Get weekly data for pivot points (previous week's data)
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # Using previous week's data to avoid look-ahead
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (will use previous week's pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    
    # 20-day average volume
    vol_ma = np.full_like(daily_volume, np.nan)
    vol_period = 20
    
    if len(daily_volume) >= vol_period:
        for i in range(vol_period, len(daily_volume)):
            vol_ma[i] = np.mean(daily_volume[i - vol_period:i])
    
    # Align daily volume MA to 6h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    
    start_idx = max(1, vol_period) + 1  # Need at least one week of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(pivot_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        # Need to get the current day's volume index
        # Since we're on 6h timeframe, we need to map to daily index
        vol_confirm = False
        if i < len(vol_ma_aligned) and not np.isnan(vol_ma_aligned[i]):
            # Get corresponding daily volume (approximate: 4x 6h bars per day)
            daily_idx = i // 4
            if daily_idx < len(daily_volume) and daily_idx < len(vol_ma) and not np.isnan(daily_volume[daily_idx]) and not np.isnan(vol_ma[daily_idx]):
                vol_confirm = daily_volume[daily_idx] > 1.5 * vol_ma[daily_idx]
        
        if vol_confirm:
            # Long: price above weekly pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.25
            # Short: price below weekly pivot
            elif close[i] < pivot_aligned[i]:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Direction_1dVolumeFilter"
timeframe = "6h"
leverage = 1.0