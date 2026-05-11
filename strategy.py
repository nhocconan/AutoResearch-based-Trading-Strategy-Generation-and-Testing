#!/usr/bin/env python3
# 6h_TwinPeak_Reversal_1wTrend
# Hypothesis: Combines weekly trend direction with twin peak (double top/bottom) reversal patterns on 6h for mean-reversion entries.
# In weekly uptrend: look for 6h double bottom near support for long entries.
# In weekly downtrend: look for 6h double top near resistance for short entries.
# Uses volume confirmation to avoid false signals.
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.
# Twin peak pattern provides high-probability reversal signals with clear invalidation levels.

name = "6h_TwinPeak_Reversal_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1w trend: higher close vs previous week ---
    close_1w = df_1w['close'].values
    weekly_up = close_1w > np.roll(close_1w, 1)
    weekly_down = close_1w < np.roll(close_1w, 1)
    weekly_up[0] = False
    weekly_down[0] = False
    
    # --- Twin peak detection on 6h ---
    # Look for double top: two similar highs with a trough in between
    # Look for double bottom: two similar lows with a peak in between
    window = 10  # lookback window for peak/trough detection
    min_distance = 5  # minimum bars between peaks
    
    # Initialize arrays
    double_top = np.zeros(n, dtype=bool)
    double_bottom = np.zeros(n, dtype=bool)
    
    for i in range(window*2 + min_distance, n):
        # Look back for two peaks/troughs
        search_start = i - window*2
        search_end = i - min_distance
        
        if search_start < 0:
            continue
            
        # Find local maxima in search window
        maxima = []
        for j in range(search_start, search_end):
            if j >= 1 and j < n-1:
                if high[j] >= high[j-1] and high[j] >= high[j+1]:
                    maxima.append((j, high[j]))
        
        # Find local minima in search window
        minima = []
        for j in range(search_start, search_end):
            if j >= 1 and j < n-1:
                if low[j] <= low[j-1] and low[j] <= low[j+1]:
                    minima.append((j, low[j]))
        
        # Check for double top: two similar highs
        if len(maxima) >= 2:
            # Take two most recent maxima
            (idx1, price1), (idx2, price2) = maxima[-2], maxima[-1]
            # Check if prices are within 1.5% of each other
            if abs(price1 - price2) / price2 < 0.015:
                # Check if there's a trough between them
                if any(m[0] > idx1 and m[0] < idx2 for m in minima):
                    double_top[i] = True
        
        # Check for double bottom: two similar lows
        if len(minima) >= 2:
            # Take two most recent minima
            (idx1, price1), (idx2, price2) = minima[-2], minima[-1]
            # Check if prices are within 1.5% of each other
            if abs(price1 - price2) / price2 < 0.015:
                # Check if there's a peak between them
                if any(m[0] > idx1 and m[0] < idx2 for m in maxima):
                    double_bottom[i] = True
    
    # --- Volume confirmation ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1w trend indicators to 6h timeframe
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up)
    weekly_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_down)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for pattern detection and volume MA
    start_idx = window*2 + min_distance + 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vol_ma[i]) or
            np.isnan(weekly_up_aligned[i]) or
            np.isnan(weekly_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend
        is_weekly_up = weekly_up_aligned[i]
        is_weekly_down = weekly_down_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if is_weekly_up and double_bottom[i] and vol_spike:
                # Long: weekly uptrend + 6h double bottom + volume spike
                signals[i] = 0.25
                position = 1
            elif is_weekly_down and double_top[i] and vol_spike:
                # Short: weekly downtrend + 6h double top + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price breaks below double bottom support or weekly trend changes
                if double_bottom[i] and low[i] < low[i-1] * 0.995 or not is_weekly_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above double top resistance or weekly trend changes
                if double_top[i] and high[i] > high[i-1] * 1.005 or not is_weekly_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals