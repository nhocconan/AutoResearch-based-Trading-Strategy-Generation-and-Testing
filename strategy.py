#!/usr/bin/env python3
# 6h_WeeklyPivot_Donchian_Breakout
# Hypothesis: Combines weekly pivot points (from 1w) with Donchian breakout (20-period on 6h) to capture institutional breakouts.
# Long when: price breaks above weekly R1 AND Donchian upper band (20) on 6h, with volume > 1.5x 20-period average.
# Short when: price breaks below weekly S1 AND Donchian lower band (20) on 6h, with volume > 1.5x 20-period average.
# Exit when price returns to weekly pivot point (PP) or Donchian middle (10-period average of high/low).
# Weekly pivots provide key institutional levels; Donchian breakouts capture momentum; volume filters avoid false breakouts.
# Works in bull markets by catching upward breaks of weekly resistance; in bear markets by catching downward breaks of weekly support.

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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly pivot points calculation ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*PP - L
    r1 = 2 * pp - low_1w
    # S1 = 2*PP - H
    s1 = 2 * pp - high_1w
    
    # --- Donchian channels (20-period) on 6h ---
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    # Middle band = (upper + lower) / 2
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian levels
        donch_upper = highest_high[i]
        donch_lower = lowest_low[i]
        donch_middle = (donch_upper + donch_lower) / 2.0
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            # Long: price breaks above weekly R1 AND Donchian upper band, with volume spike
            if close[i] > r1_aligned[i] and close[i] > donch_upper and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 AND Donchian lower band, with volume spike
            elif close[i] < s1_aligned[i] and close[i] < donch_lower and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price returns to weekly pivot OR Donchian middle
                if close[i] <= pp_aligned[i] or close[i] <= donch_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly pivot OR Donchian middle
                if close[i] >= pp_aligned[i] or close[i] >= donch_middle:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals