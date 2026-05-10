#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_1dTrend_Volume
Hypothesis: Price retests of Camarilla R1/S1 levels with 1d trend alignment and volume confirmation provide high-probability mean-reversion entries in both bull and bear markets. R1/S1 are key intraday pivot levels where reversals often occur. Using 1d EMA34 for trend filter ensures trades align with higher timeframe momentum, while volume filter confirms institutional interest. Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and enough data for Camarilla
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend filter
        uptrend_1d = close[i] > ema34_1d_aligned[i]
        downtrend_1d = close[i] < ema34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_avg = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_filter = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long entry: price near S1 (within 0.1%) + uptrend + volume
            if (abs(close[i] - camarilla_S1_aligned[i]) / camarilla_S1_aligned[i] < 0.001 and
                uptrend_1d and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price near R1 (within 0.1%) + downtrend + volume
            elif (abs(close[i] - camarilla_R1_aligned[i]) / camarilla_R1_aligned[i] < 0.001 and
                  downtrend_1d and volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price touches R1 or trend fails
            if (close[i] >= camarilla_R1_aligned[i] or not uptrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches S1 or trend fails
            if (close[i] <= camarilla_S1_aligned[i] or not downtrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals