#!/usr/bin/env python3
"""
4H_MR_Camarilla_R1_S1_Bounce_1D_Trend_Volume
Hypothesis: Fade extreme moves at 1d Camarilla R1/S1 levels in trending 4h markets (EMA50).
Long when price touches 1d S1 and 4h EMA50 is rising; short when price touches 1d R1 and 4h EMA50 is falling.
Volume confirmation: current volume > 1.3x 20-period average volume.
Mean reversion at strong support/resistance with trend filter works in both bull and bear markets.
Target: 20-40 trades/year to minimize fee drag.
"""
name = "4H_MR_Camarilla_R1_S1_Bounce_1D_Trend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = pd.Series(df_4h['close'])
    ema_4h_50 = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Get 1d data for Camarilla levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_4h_50_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 4h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price touches S1 and EMA50 is rising (current > previous)
            if (low[i] <= s1_aligned[i] and 
                ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price touches R1 and EMA50 is falling (current < previous)
            elif (high[i] >= r1_aligned[i] and 
                  ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or EMA50 crosses
            if position == 1 and (high[i] >= r1_aligned[i] or ema_4h_50_aligned[i] < ema_4h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (low[i] <= s1_aligned[i] or ema_4h_50_aligned[i] > ema_4h_50_aligned[i-1]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals