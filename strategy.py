#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_Filter
Hypothesis: Uses 1-week Elder Ray (Bull/Bear Power) to determine market direction and 6-hour EMA13 pullbacks for entry. Bull Power > 0 and Bear Power < 0 from weekly timeframe defines trend, with entries on pullbacks to EMA13 in the direction of the weekly trend. Designed for low trade frequency (12-37/year) to minimize fee drift while capturing swing moves in both bull and bear markets. Weekly trend filter avoids whipsaws during sideways periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Elder Ray (requires 13 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 13-period EMA on weekly close
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1w['high'].values - ema13_1w
    bear_power = df_1w['low'].values - ema13_1w
    
    # Determine weekly trend: Bull Power > 0 and Bear Power < 0 = uptrend
    # Bear Power > 0 and Bull Power < 0 = downtrend (both conditions rarely true together)
    # More practical: Bull Power > 0 = uptrend bias, Bear Power < 0 = downtrend bias
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Get 6h EMA13 for pullback entries
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Wait for EMA13 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema13_6h[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filters from Elder Ray
        weekly_uptrend = bull_power_aligned[i] > 0  # Weekly highs above EMA13
        weekly_downtrend = bear_power_aligned[i] < 0  # Weekly lows below EMA13
        
        # 6h price relative to EMA13 for pullback entries
        price_above_ema = close[i] > ema13_6h[i]
        price_below_ema = close[i] < ema13_6h[i]
        
        # Entry conditions: pullback to EMA13 in direction of weekly trend
        long_entry = weekly_uptrend and price_below_ema  # Pullback in uptrend
        short_entry = weekly_downtrend and price_above_ema  # Pullback in downtrend
        
        # Exit conditions: price crosses EMA13 against the trend
        long_exit = not weekly_uptrend or price_above_ema  # Trend broken or price above EMA
        short_exit = not weekly_downtrend or price_below_ema  # Trend broken or price below EMA
        
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

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0