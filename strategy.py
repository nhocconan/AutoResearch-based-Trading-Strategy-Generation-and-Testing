#!/usr/bin/env python3
"""
6h_weekly_pivot_breakout_v1
Hypothesis: Weekly pivot levels (from Monday open) act as strong support/resistance on 6h timeframe.
Breakout above weekly R1 with volume confirmation = long; breakdown below weekly S1 = short.
Works in both bull and bear markets because weekly pivots adapt to price action and
breakouts capture momentum in trending phases while avoiding false breakouts in ranges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation (weekly OHLC from Monday open)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly pivot points (standard)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * pivot - low_1w
    weekly_s1 = 2 * pivot - high_1w
    
    # Align weekly levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S1
            if close[i] < s1_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R1
            if close[i] > r1_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long: price breaks above weekly R1 with volume
            if (close[i] > r1_6h[i] and 
                vol_confirm):
                position = 1
                signals[i] = 0.25
            # Breakdown short: price breaks below weekly S1 with volume
            elif (close[i] < s1_6h[i] and 
                  vol_confirm):
                position = -1
                signals[i] = -0.25
    
    return signals