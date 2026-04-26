#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation (>1.5x median).
Enters long when price breaks above R1 with volume confirmation and bullish 1d trend (price > EMA50).
Enters short when price breaks below S1 with volume confirmation and bearish 1d trend (price < EMA50).
Exits on opposite breakout or when price returns to Camarilla pivot (close level).
Uses discrete position sizing (0.25) to minimize churn. Target: 50-150 trades over 4 years.
Works in both bull and bear markets by following 1d trend filter.
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
    
    # Calculate Camarilla levels for 12h (based on previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Previous bar's values for level calculation (to avoid look-ahead)
    h_12h_prev = np.roll(h_12h, 1)
    l_12h_prev = np.roll(l_12h, 1)
    c_12h_prev = np.roll(c_12h, 1)
    h_12h_prev[0] = np.nan
    l_12h_prev[0] = np.nan
    c_12h_prev[0] = np.nan
    
    # Calculate Camarilla R1 and S1 levels
    rng_12h = h_12h_prev - l_12h_prev
    r1_12h = c_12h_prev + (rng_12h * 1.1 / 12)
    s1_12h = c_12h_prev - (rng_12h * 1.1 / 12)
    
    # Align to 12h primary timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # Volume confirmation: volume > 1.5x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Load 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 50-period volume median, 50-period EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above R1 + volume confirmation + bullish 1d trend
        if close[i] > r1_12h_aligned[i] and volume_confirm[i] and close[i] > ema_50_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 + volume confirmation + bearish 1d trend
        elif close[i] < s1_12h_aligned[i] and volume_confirm[i] and close[i] < ema_50_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite breakout (price returns to the other level)
        elif position == 1 and close[i] < s1_12h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r1_12h_aligned[i]:
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0