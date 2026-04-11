#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot levels from 1d with volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance in 12h timeframe.
# Price reverses at L3/H3 levels with volume confirmation. Works in both bull/bear markets
# as it captures mean reversion at key institutional levels. Low frequency (~20-30/year)
# to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    H4 = pivot + (range_1d * 1.1 / 2)
    H3 = pivot + (range_1d * 1.1 / 4)
    H2 = pivot + (range_1d * 1.1 / 6)
    H1 = pivot + (range_1d * 1.1 / 12)
    L1 = pivot - (range_1d * 1.1 / 12)
    L2 = pivot - (range_1d * 1.1 / 6)
    L3 = pivot - (range_1d * 1.1 / 4)
    L4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (previous day's levels)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: price at Camarilla H3/L3 with volume confirmation
        if (close[i] <= H3_aligned[i] and close[i] >= H3_aligned[i] * 0.999 and  # Near H3 resistance
            vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        elif (close[i] >= L3_aligned[i] and close[i] <= L3_aligned[i] * 1.001 and  # Near L3 support
              vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Exit: price moves away from pivot level
        elif position == 1 and close[i] < L3_aligned[i] * 0.995:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > H3_aligned[i] * 1.005:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals