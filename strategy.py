#!/usr/bin/env python3
# 12H_1W_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend filter (1w EMA50) with 12h Camarilla R1/S1 breakouts.
# Weekly trend provides robust trend filtering for 12h timeframe, reducing whipsaws in sideways markets.
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear markets
# by aligning with higher timeframe trend. Target: 50-150 total trades over 4 years (12-37/year).

name = "12H_1W_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = pivot + range_ * 1.1 / 4  # R1 = pivot + (range * 1.1 / 4)
    s1 = pivot - range_ * 1.1 / 4  # S1 = pivot - (range * 1.1 / 4)
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above weekly EMA50 + volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below weekly EMA50 + volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly EMA50 (trend change)
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly EMA50 (trend change)
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals