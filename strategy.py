#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume confirmation (>1.5x median). 
Enters long when price breaks above R1 with volume confirmation and bullish weekly trend. 
Enters short when price breaks below S1 with volume confirmation and bearish weekly trend. 
Exits on opposite breakout. Uses discrete position sizing (0.25) to minimize churn. 
Target: 30-100 trades over 4 years (7-25/year). Works in both bull and bear markets by following 
weekly trend filter and avoiding excessive whipsaws via volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 1d (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_1d_prev = np.roll(h_1d, 1)
    l_1d_prev = np.roll(l_1d, 1)
    c_1d_prev = np.roll(c_1d, 1)
    h_1d_prev[0] = np.nan
    l_1d_prev[0] = np.nan
    c_1d_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_1d = h_1d_prev - l_1d_prev
    r1_1d = c_1d_prev + (rng_1d * 1.1 / 12)
    s1_1d = c_1d_prev - (rng_1d * 1.1 / 12)
    
    # Align to 1d primary timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: volume > 1.5x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Load weekly data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 34-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_34_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R1 + volume confirmation + bullish weekly trend
        if close[i] > r1_1d_aligned[i] and volume_confirm[i] and close[i] > ema_34_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume confirmation + bearish weekly trend
        elif close[i] < s1_1d_aligned[i] and volume_confirm[i] and close[i] < ema_34_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < s1_1d_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_1d_aligned[i]:
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

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0