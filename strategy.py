#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1
Hypothesis: Uses Camarilla pivot levels (R1/S1) from 1d timeframe for entry, filtered by 1d EMA34 trend and volume spike.
In bull markets, price breaks above R1 in uptrend; in bear markets, price breaks below S1 in downtrend.
Volume spike confirms institutional participation. Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1D Data for Camarilla Pivots and Trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    for i in range(n):
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                ema34_1d_aligned[i] < close[i] and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  ema34_1d_aligned[i] > close[i] and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below EMA34 or volume drops
            if (close[i] < ema34_1d_aligned[i] or 
                vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above EMA34 or volume drops
            if (close[i] > ema34_1d_aligned[i] or 
                vol_ratio[i] < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals