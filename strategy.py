#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_VolumeFilter_V1
Hypothesis: Combine weekly pivot levels with 4-hour Donchian breakout and volume confirmation to capture institutional-level breakouts in both bull and bear markets.
Weekly pivots provide strong support/resistance from longer-term structure, while 4h Donchian captures intermediate momentum.
Volume filter ensures breakouts have participation. Target: 20-50 trades/year for low friction.
Works in bull/bear: breakouts capture momentum; weekly pivot context prevents counter-trend entries.
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
    volume = prices['volume'].values
    
    # === 4h data for Donchian channel ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian(20) on 4h: upper/lower bounds
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h
    donch_high_6h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_6h = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === Weekly data for pivot points ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard)
    pp = (high_1w + low_1w + close_1w) / 3.0
    range_hl = high_1w - low_1w
    
    r1 = pp + (range_hl * 1.0 / 3.0)  # R1
    s1 = pp - (range_hl * 1.0 / 3.0)  # S1
    r2 = pp + range_hl                # R2
    s2 = pp - range_hl                # S2
    
    # Align weekly pivots to 6h
    pp_6h = align_htf_to_ltf(prices, df_1w, pp)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    
    # === Volume confirmation ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-period Donchian and volume average
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_6h[i]) or np.isnan(donch_low_6h[i]) or
            np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = volume[i] > 1.3 * vol_avg_20[i]
        
        # Entry: only when flat
        if position == 0:
            # Long: break above Donchian high AND above weekly R1 with volume
            if close[i] > donch_high_6h[i] and close[i] > r1_6h[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low AND below weekly S1 with volume
            elif close[i] < donch_low_6h[i] and close[i] < s1_6h[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit: reverse signal or opposite pivot breach
        elif position == 1:
            # Exit long if price breaks below weekly S1 or Donchian low
            if close[i] < s1_6h[i] or close[i] < donch_low_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above weekly R1 or Donchian high
            if close[i] > r1_6h[i] or close[i] > donch_high_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0