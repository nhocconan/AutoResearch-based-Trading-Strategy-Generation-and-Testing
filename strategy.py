#!/usr/bin/env python3
# 6h_WeeklyPivot_Donchian_Breakout
# Hypothesis: Combines weekly pivot levels (from 1w) with Donchian(20) breakout on 6h.
# Long when: price breaks above Donchian high(20) AND price is above weekly pivot (support/resistance).
# Short when: price breaks below Donchian low(20) AND price is below weekly pivot.
# Exit when price crosses back to Donchian midpoint or crosses back across weekly pivot.
# Weekly pivot provides structural bias from higher timeframe, reducing false breakouts.
# Works in bull markets by buying breakouts above weekly pivot and in bear by selling breakdowns below weekly pivot.
# Donchian gives clear entry/exit levels, weekly pivot filters counter-trend noise.

name = "6h_WeeklyPivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Donchian(20) channels ---
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # --- Weekly Pivot Levels (using prior week's OHLC) ---
    # Calculate pivot points: P = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)
    s1 = np.full(len(close_1w), np.nan)
    
    for i in range(1, len(close_1w)):
        pivot[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3
        r1[i] = 2 * pivot[i] - low_1w[i-1]
        s1[i] = 2 * pivot[i] - high_1w[i-1]
    
    # Align weekly pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Donchian(20) and weekly pivot
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        if position == 0:
            if breakout_up and close[i] > pivot_aligned[i]:
                # Long: upward breakout above weekly pivot
                signals[i] = 0.25
                position = 1
            elif breakout_down and close[i] < pivot_aligned[i]:
                # Short: downward breakout below weekly pivot
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls to midpoint OR crosses below weekly pivot
                if close[i] < donchian_mid[i] or close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises to midpoint OR crosses above weekly pivot
                if close[i] > donchian_mid[i] or close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals