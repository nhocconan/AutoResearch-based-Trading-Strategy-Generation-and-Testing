#!/usr/bin/env python3
# 1D_Weekly_Triangle_Breakout
# Hypothesis: Weekly triangle patterns (ascending/descending) identified by higher lows/lower highs
# provide high-probability breakout signals. Breakout in direction of trend with volume confirmation
# captures sustained moves. Works in bull markets (ascending triangle breakouts) and bear markets
# (descending triangle breakdowns) by following the weekly trend. Low trade frequency expected due to
# strict triangle formation requirements + trend filter + volume confirmation.

name = "1D_Weekly_Triangle_Breakout"
timeframe = "1d"
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
    
    # 1w data for triangle pattern and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Triangle pattern detection: higher lows and lower highs over 3 weeks
    # For ascending triangle: higher lows + roughly equal highs
    # For descending triangle: lower highs + roughly equal lows
    # We'll use 3-week lookback for simplicity
    
    # Calculate weekly swing points
    def find_swing_points(arr, window=2):
        """Find local highs and lows"""
        highs = np.full_like(arr, np.nan)
        lows = np.full_like(arr, np.nan)
        for i in range(window, len(arr) - window):
            if arr[i] == np.max(arr[i-window:i+window+1]):
                highs[i] = arr[i]
            if arr[i] == np.min(arr[i-window:i+window+1]):
                lows[i] = arr[i]
        return highs, lows
    
    highs_1w, lows_1w = find_swing_points(close_1w, 2)
    
    # Get recent swing points (last 3 weeks)
    def get_recent_swing(arr, lookback=3):
        """Get non-NaN values from recent lookback period"""
        start_idx = max(0, len(arr) - lookback*5)  # Approximate weeks
        subset = arr[start_idx:]
        valid_vals = subset[~np.isnan(subset)]
        if len(valid_vals) >= 2:
            return valid_vals[-2:]  # Last two points
        return np.array([])
    
    # Triangle conditions
    def is_ascending_triangle(lows, highs):
        """Check for ascending triangle: higher lows, relatively flat highs"""
        if len(lows) < 2 or len(highs) < 2:
            return False
        # Check if lows are rising
        lows_rising = lows[-1] > lows[-2]
        # Check if highs are relatively flat (within 2%)
        highs_flat = abs(highs[-1] - highs[-2]) / highs[-2] < 0.02
        return lows_rising and highs_flat
    
    def is_descending_triangle(lows, highs):
        """Check for descending triangle: lower highs, relatively flat lows"""
        if len(lows) < 2 or len(highs) < 2:
            return False
        # Check if highs are falling
        highs_falling = highs[-1] < highs[-2]
        # Check if lows are relatively flat (within 2%)
        lows_flat = abs(lows[-1] - lows[-2]) / lows[-2] < 0.02
        return highs_falling and lows_flat
    
    # Pre-calculate triangle signals for each week
    asc_triangle = np.full(len(close_1w), False)
    desc_triangle = np.full(len(close_1w), False)
    
    for i in range(10, len(close_1w)):  # Start after enough data for swing detection
        # Get swing points up to current week
        recent_highs = highs_1w[max(0, i-10):i+1]
        recent_lows = lows_1w[max(0, i-10):i+1]
        
        # Filter out NaN values
        recent_highs = recent_highs[~np.isnan(recent_highs)]
        recent_lows = recent_lows[~np.isnan(recent_lows)]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            asc_triangle[i] = is_ascending_triangle(recent_lows, recent_highs)
            desc_triangle[i] = is_descending_triangle(recent_lows, recent_highs)
    
    # Align triangle signals to daily timeframe
    asc_triangle_aligned = align_htf_to_ltf(prices, df_1w, asc_triangle.astype(float))
    desc_triangle_aligned = align_htf_to_ltf(prices, df_1w, desc_triangle.astype(float))
    
    # Weekly trend filter: EMA20
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation (20-day average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough history for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(asc_triangle_aligned[i]) or np.isnan(desc_triangle_aligned[i]) or \
           np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ascending triangle breakout above weekly EMA20 with volume
            if asc_triangle_aligned[i] > 0.5 and close[i] > ema_20_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: descending triangle breakdown below weekly EMA20 with volume
            elif desc_triangle_aligned[i] > 0.5 and close[i] < ema_20_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: breakdown below weekly EMA20 or descending triangle forms
            if close[i] < ema_20_aligned[i] or desc_triangle_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: breakout above weekly EMA20 or ascending triangle forms
            if close[i] > ema_20_aligned[i] or asc_triangle_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals