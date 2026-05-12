#!/usr/bin/env python3
# 4h_WeeklyPivot_Volume_Trend
# Hypothesis: Weekly pivot points act as strong support/resistance levels.
# In trending markets, price respects these levels as pullback entries.
# In ranging markets, reversals occur at these levels.
# Volume confirmation filters false breakouts.
# Timeframe: 4h balances trade frequency (~20-50/year) with signal quality.
# Works in bull/bear: uses price action at key levels rather than trend direction.

name = "4h_WeeklyPivot_Volume_Trend"
timeframe = "4h"
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
    
    # === WEEKLY DATA FOR PIVOT LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly levels to 4h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)  # Moderate volume filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # For volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price bounces off S1 or S2 with volume
            if ((close[i] > s1_1w_aligned[i] and low[i] <= s1_1w_aligned[i] * 1.005) or
                (close[i] > s2_1w_aligned[i] and low[i] <= s2_1w_aligned[i] * 1.005)) and \
               volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects at R1 or R2 with volume
            elif ((close[i] < r1_1w_aligned[i] and high[i] >= r1_1w_aligned[i] * 0.995) or
                  (close[i] < r2_1w_aligned[i] and high[i] >= r2_1w_aligned[i] * 0.995)) and \
                 volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S2 or reaches R2
            if close[i] < s2_1w_aligned[i] or close[i] > r2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R2 or reaches S2
            if close[i] > r2_1w_aligned[i] or close[i] < s2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals