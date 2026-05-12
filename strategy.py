#!/usr/bin/env python3
# 6H_DONCHIAN_BREAKOUT_WEEKLY_PIVOT_DIRECTION
# Hypothesis: In 6h timeframe, use weekly pivot points to determine market direction (above/below weekly pivot).
# Go long when price breaks above Donchian(20) high and weekly trend is up (price > weekly pivot).
# Go short when price breaks below Donchian(20) low and weekly trend is down (price < weekly pivot).
# Weekly pivot acts as a trend filter to avoid counter-trend trades, while Donchian breakout captures momentum.
# Works in both bull and bear markets: weekly pivot defines trend, Donchian breakout provides entry signal.
# Target: 15-25 trades/year on 6h timeframe.

name = "6H_DONCHIAN_BREAKOUT_WEEKLY_PIVOT_DIRECTION"
timeframe = "6h"
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
    
    # Weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (High + Low + Close) / 3
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels (20-period) on 6x timeframe
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > weekly pivot (uptrend) + break above Donchian high
            if (close[i] > weekly_pivot_aligned[i] and 
                high[i] > highest_high[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < weekly pivot (downtrend) + break below Donchian low
            elif (close[i] < weekly_pivot_aligned[i] and 
                  low[i] < lowest_low[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or breakdown below Donchian low
            if (close[i] <= weekly_pivot_aligned[i] or 
                low[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or break above Donchian high
            if (close[i] >= weekly_pivot_aligned[i] or 
                high[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals