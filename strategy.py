#!/usr/bin/env python3
# 6h_6H_WeeklyPivot_DonchianBreakout_TrendFilter
# Hypothesis: Donchian(20) breakout on 6h with weekly pivot trend filter.
# Long when: price > Donchian upper + price > weekly pivot pivot point.
# Short when: price < Donchian lower + price < weekly pivot pivot point.
# Weekly pivot provides higher timeframe bias (bullish/bearish) to filter breakouts.
# Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in both bull and bear markets by following weekly pivot trend.

name = "6h_6H_WeeklyPivot_DonchianBreakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    ph = np.concatenate([[high_1w[0]], high_1w[:-1]])  # previous high
    pl = np.concatenate([[low_1w[0]], low_1w[:-1]])   # previous low
    pc = np.concatenate([[close_1w[0]], close_1w[:-1]]) # previous close
    
    # Calculate weekly pivot point (PP) = (H + L + C) / 3
    pp = (ph + pl + pc) / 3.0
    
    # Align weekly pivot to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Calculate Donchian(20) on 6h
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    
    if len(high) >= 20:
        for i in range(20-1, len(high)):
            donch_high[i] = np.max(high[i-20+1:i+1])
            donch_low[i] = np.min(low[i-20+1:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20 - 1  # Donchian period
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above weekly pivot
            if close[i] > donch_high[i] and close[i] > pp_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND below weekly pivot
            elif close[i] < donch_low[i] and close[i] < pp_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR below weekly pivot
            if close[i] < donch_low[i] or close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR above weekly pivot
            if close[i] > donch_high[i] or close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals