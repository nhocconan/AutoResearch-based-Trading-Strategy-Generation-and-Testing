#!/usr/bin/env python3
"""
4h_WeeklyPivot_R1_S1_Breakout_VolumeTrend_v3
Hypothesis: Weekly pivot points (R1/S1) with volume confirmation and 1d EMA50 trend filter. Only takes long above R1 in uptrend (close>EMA50) and short below S1 in downtrend (close<EMA50). Reduces whipsaw by requiring trend alignment. Target: 20-40 trades/year to minimize fee drag.
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
    
    # === 1d data for EMA50 trend filter and volume average ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average (20-period) for confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # === Weekly data for Pivot points (R1, S1) ===
    df_1w = get_htf_data(prices, '1w')
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = (2 * Pivot) - L
    # S1 = (2 * Pivot) - H
    pivot_w = (high_w + low_w + close_w) / 3
    R1_w = (2 * pivot_w) - low_w
    S1_w = (2 * pivot_w) - high_w
    
    # Align Weekly Pivot levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1_w)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1_w)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA50 and volume average
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Entry conditions: require trend alignment and close beyond pivot level
        if position == 0:
            # Long: price closes above weekly R1 + above 1d EMA50 + volume
            if close[i] > R1_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price closes below weekly S1 + below 1d EMA50 + volume
            elif close[i] < S1_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price returns to opposite weekly pivot level
        elif position == 1:
            if close[i] < S1_aligned[i]:  # exit long when price closes below S1
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > R1_aligned[i]:  # exit short when price closes above R1
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WeeklyPivot_R1_S1_Breakout_VolumeTrend_v3"
timeframe = "4h"
leverage = 1.0