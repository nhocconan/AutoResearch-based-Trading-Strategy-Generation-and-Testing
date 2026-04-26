#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: 12h Camarilla R1/S1 level breakout with weekly EMA20 trend filter and volume spike (>1.8x 24-period median). 
Enters long when price closes above R1 with volume confirmation and bullish weekly trend. 
Enters short when price closes below S1 with volume confirmation and bearish weekly trend. 
Exits on opposite Camarilla level touch (long exits at S1, short exits at R1). 
Uses discrete position sizing (0.25) to minimize churn. Target: 50-150 trades over 4 years. 
Uses weekly trend to avoid bear market whipsaw, volume filter ensures conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.8x 24-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=24, min_periods=24).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    # Load weekly data for HTF trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Load daily data for Camarilla calculation (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 24-period volume median, 20-period weekly EMA)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: close above R1 + volume confirm + bullish weekly trend
        if close[i] > camarilla_r1_aligned[i] and volume_confirm[i] and close[i] > ema20_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: close below S1 + volume confirm + bearish weekly trend
        elif close[i] < camarilla_s1_aligned[i] and volume_confirm[i] and close[i] < ema20_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price touches S1, short exits when price touches R1
        elif position == 1 and close[i] <= camarilla_s1_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= camarilla_r1_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0