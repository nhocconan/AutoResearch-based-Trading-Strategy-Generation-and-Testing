#!/usr/bin/env python3
"""
6h_DailyPivot_R3S3_Fade_R4S4_Breakout
Hypothesis: Fade at daily R3/S3 levels with mean reversion, breakout continuation at R4/S4 levels.
Uses 1-week trend filter to avoid counter-trend trades. Designed for low frequency (15-30 trades/year)
with strong performance in both bull and bear markets by combining mean reversion at extreme levels
with trend-following breakouts at stronger levels.
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
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels (R3, S3, R4, S4 from prior day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3 = np.full(len(close_1d), np.nan)  # R3 level
    s3 = np.full(len(close_1d), np.nan)  # S3 level
    r4 = np.full(len(close_1d), np.nan)  # R4 level
    s4 = np.full(len(close_1d), np.nan)  # S4 level
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        diff = ph - pl
        r3[i] = pc + 1.1 * diff  # R3
        s3[i] = pc - 1.1 * diff  # S3
        r4[i] = pc + 1.5 * diff  # R4
        s4[i] = pc - 1.5 * diff  # S4
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema50_1w[i] = close_1w[i] * alpha + ema50_1w[i-1] * (1 - alpha)
    
    # Align all levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long fade at S3: price < S3 with volume filter and above weekly trend
            if (close[i] < s3_aligned[i] and vol_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short fade at R3: price > R3 with volume filter and below weekly trend
            elif (close[i] > r3_aligned[i] and vol_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            # Long breakout at R4: price > R4 with volume filter and above weekly trend
            elif (close[i] > r4_aligned[i] and vol_filter[i] and 
                  close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout at S4: price < S4 with volume filter and below weekly trend
            elif (close[i] < s4_aligned[i] and vol_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below S3 or weekly trend turns down
            if (close[i] > s3_aligned[i] or close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above R3 or weekly trend turns up
            if (close[i] < r3_aligned[i] or close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_DailyPivot_R3S3_Fade_R4S4_Breakout"
timeframe = "6h"
leverage = 1.0