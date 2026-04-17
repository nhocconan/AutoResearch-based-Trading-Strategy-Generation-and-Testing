#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_VolumeFilter_V1
Hypothesis: On 6h timeframe, buy when price breaks above weekly Donchian(20) high with price above weekly pivot point and volume > 1.5x weekly average; sell when price breaks below weekly Donchian(20) low with price below weekly pivot and volume > 1.5x weekly average. Uses weekly structure for trend direction and volume confirmation to filter false breakouts, designed for low trade frequency (12-37/year) to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point: P = (H+L+C)/3"""
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Data (HTF for Donchian, pivot, volume) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    upper_1w, lower_1w = calculate_donchian_channels(high_1w, low_1w, 20)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    
    # Weekly pivot point
    pivot_1w = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Weekly average volume (20-period)
    vol_avg_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_avg_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_1w_aligned[i]) or 
            np.isnan(lower_1w_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(vol_avg_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current weekly bar's volume for confirmation
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        
        # Volume filter: current volume > 1.5x weekly average volume
        vol_filter = vol_1w_current > 1.5 * vol_avg_1w_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly Donchian upper with price above weekly pivot and volume filter
            if close[i] > upper_1w_aligned[i] and close[i] > pivot_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian lower with price below weekly pivot and volume filter
            elif close[i] < lower_1w_aligned[i] and close[i] < pivot_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price breaks below weekly Donchian lower (opposite breakout)
            if close[i] < lower_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price breaks above weekly Donchian upper (opposite breakout)
            if close[i] > upper_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0