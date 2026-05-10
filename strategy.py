#!/usr/bin/env python3
# 6h_Williams_Alligator_WeeklyTrend_Filter
# Hypothesis: The Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
# In strong trends (Alligator aligned and mouth open), we trade pullbacks to the Teeth (SMMA8).
# Weekly trend filter (price vs 50-week SMA) ensures we only trade in the direction of the higher-timeframe trend.
# This avoids counter-trend trades and works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Williams Alligator uses SMMA (Smoothed Moving Average) which is less reactive than EMA/SMA, reducing whipsaws.
# Weekly timeframe provides strong trend filter for 6h chart, suitable for 6-12 month trends.

name = "6h_Williams_Alligator_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Williams Alligator on 6h chart: Jaw (SMMA13), Teeth (SMMA8), Lips (SMMA5)
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need all SMMA values (13, 8, 5) and weekly SMA50
    start_idx = max(13, 8, 5)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Alligator alignment and direction
        # Alligator is aligned when Jaw > Teeth > Lips (downtrend) or Lips > Teeth > Jaw (uptrend)
        aligned_down = jaw[i] > teeth[i] and teeth[i] > lips[i]
        aligned_up = lips[i] > teeth[i] and teeth[i] > jaw[i]
        
        # Mouth open: gap between Jaw and Lips
        mouth_open = abs(jaw[i] - lips[i]) > (teeth[i] * 0.001)  # 0.1% of teeth value
        
        # Price position relative to Teeth (SMMA8) for pullback entries
        if i > 0:
            cross_above_teeth = (close[i] > teeth[i]) and (close[i-1] <= teeth[i-1])
            cross_below_teeth = (close[i] < teeth[i]) and (close[i-1] >= teeth[i-1])
        else:
            cross_above_teeth = False
            cross_below_teeth = False
        
        if position == 0:
            # Long entry: weekly uptrend + Alligator aligned up + mouth open + pullback to Teeth from below
            if weekly_uptrend and aligned_up and mouth_open and cross_above_teeth:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + Alligator aligned down + mouth open + pullback to Teeth from above
            elif weekly_downtrend and aligned_down and mouth_open and cross_below_teeth:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend breaks, Alligator misaligns, or reverse signal
            if (not weekly_uptrend) or (not aligned_up) or (not mouth_open) or cross_below_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend breaks, Alligator misaligns, or reverse signal
            if (not weekly_downtrend) or (not aligned_down) or (not mouth_open) or cross_above_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals