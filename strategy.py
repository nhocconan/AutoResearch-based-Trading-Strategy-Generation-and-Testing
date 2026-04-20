#!/usr/bin/env python3
"""
4h_Pivot_R1_S1_Breakout_Volume_Conservative_v2
Hypothesis: Use daily Camarilla pivot levels (R1/S1) as support/resistance. Enter long when price breaks above R1 with volume confirmation, short when breaks below S1 with volume confirmation. Exit when price returns to the pivot point (PP). Uses 4h timeframe with 12h trend filter to avoid counter-trend trades. Designed for fewer trades (<400 total) to minimize fee drag and work in both bull and bear markets via mean reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Pivot_R1_S1_Breakout_Volume_Conservative_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 4h: use previous day's levels (already closed)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h trend filter (EMA25)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    ema25_12h = pd.Series(df_12h['close'].values).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is valid
    
    for i in range(start_idx, n):
        # Skip if any data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema25_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, and 12h trend is up (price > EMA25)
            if close[i] > r1_aligned[i] and vol_ok and close[i] > ema25_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, and 12h trend is down (price < EMA25)
            elif close[i] < s1_aligned[i] and vol_ok and close[i] < ema25_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals