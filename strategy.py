#!/usr/bin/env python3
name = "6h_DonchianBreakout_WeeklyPivot_TrendFilter"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Weekly pivot points (from previous week)
    prev_high_w = df_w['high'].shift(1).values
    prev_low_w = df_w['low'].shift(1).values
    prev_close_w = df_w['close'].shift(1).values
    
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    r4_w = pivot_w + 3 * (prev_high_w - prev_low_w)  # R4 = P + 3*(H-L)
    s4_w = pivot_w - 3 * (prev_high_w - prev_low_w)  # S4 = P - 3*(H-L)
    
    # Align weekly pivot levels to 6h
    pivot_w_aligned = align_htf_to_ltf(prices, df_w, pivot_w)
    r4_w_aligned = align_htf_to_ltf(prices, df_w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_w, s4_w)
    
    # Donchian(20) channels on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for volume MA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(r4_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with weekly bullish bias (price > weekly pivot) and volume
            if (high[i] > highest_high[i] and close[i] > pivot_w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with weekly bearish bias (price < weekly pivot) and volume
            elif (low[i] < lowest_low[i] and close[i] < pivot_w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below Donchian low or weekly bearish flip
            if close[i] < lowest_low[i] or close[i] < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above Donchian high or weekly bullish flip
            if close[i] > highest_high[i] or close[i] > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakout with weekly pivot bias and volume confirmation.
# Weekly pivot provides institutional reference; R4/S4 levels act as strong support/resistance.
# Donchian breakout captures momentum; weekly pivot bias ensures alignment with higher timeframe structure.
# Volume filter confirms institutional participation. Designed for 50-150 total trades over 4 years.
# Works in bull markets (breakouts with upward bias) and bear markets (breakdowns with downward bias).