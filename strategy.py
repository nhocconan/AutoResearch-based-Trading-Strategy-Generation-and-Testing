#!/usr/bin/env python3
# 6h_Donchian_WeeklyPivot_Volume
# Hypothesis: Use 6h Donchian(20) breakout confirmed by weekly pivot direction (from 1w data)
# and volume spike. Enter long when price breaks above Donchian upper band and weekly pivot
# shows bullish bias (price above weekly pivot point). Enter short when price breaks below
# Donchian lower band and weekly pivot shows bearish bias (price below weekly pivot point).
# Weekly pivot provides multi-day context to filter false breakouts, working in both bull
# (catch breakouts in uptrend) and bear (catch breakdowns in downtrend) markets.
# Volume spike ensures breakouts have conviction. Target: 12-37 trades/year.

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """
    Calculate weekly pivot point and support/resistance levels.
    Pivot Point (PP) = (High + Low + Close) / 3
    Returns pivot point array.
    """
    return (high + low + close) / 3.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point
    weekly_pivot = calculate_weekly_pivot(high_1w, low_1w, close_1w)
    
    # Donchian channels on 6h (20-period)
    period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < period:
            highest_high[i] = np.max(high[max(0, i-period+1):i+1])
            lowest_low[i] = np.min(low[max(0, i-period+1):i+1])
        else:
            highest_high[i] = max(highest_high[i-1], high[i])
            lowest_low[i] = min(lowest_low[i-1], low[i])
            if i >= period:
                highest_high[i] = max(highest_high[i], high[i-period+1])
                lowest_low[i] = min(lowest_low[i], low[i-period+1])
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian period and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Weekly pivot bias
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Donchian breakout up + price above weekly pivot + volume
            if breakout_up and price_above_pivot and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Donchian breakout down + price below weekly pivot + volume
            elif breakout_down and price_below_pivot and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel or pivot bias fails
            if close[i] < highest_high[i] or not price_above_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel or pivot bias fails
            if close[i] > lowest_low[i] or not price_below_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals