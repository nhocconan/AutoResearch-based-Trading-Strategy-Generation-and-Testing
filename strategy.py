#!/usr/bin/env python3
"""
1d_WilliamsAlligator_WeeklyTrend_1wTrendFilter
Hypothesis: Uses Williams Alligator (SMAs of median price) on daily timeframe with weekly trend filter.
Long when price > Alligator Jaw and weekly trend up, short when price < Alligator Jaw and weekly trend down.
Designed to capture trends in both bull and bear markets with low trade frequency.
"""

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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator lines (13, 8, 5 periods SMAs with future shifts)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend
    weekly_close = df_1w['close'].values
    weekly_ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_ema21_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(weekly_ema21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: Mouth closed (all lines intertwined) = no trend
        # Mouth open (lips > teeth > jaw) = uptrend, reverse = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Weekly trend filter
        weekly_up = close[i] > weekly_ema21_aligned[i]
        weekly_down = close[i] < weekly_ema21_aligned[i]
        
        # Entry conditions: Alligator aligned with weekly trend
        long_entry = alligator_long and weekly_up
        short_entry = alligator_short and weekly_down
        
        # Exit: When Alligator reverses or weekly trend changes
        long_exit = not alligator_long or not weekly_up
        short_exit = not alligator_short or not weekly_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsAlligator_WeeklyTrend_1wTrendFilter"
timeframe = "1d"
leverage = 1.0