#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_WeeklyTrend
Hypothesis: 6-hour Camarilla pivot breakouts with weekly trend filter to capture directional moves in both bull and bear markets.
Uses R1/S1 breakouts in direction of weekly EMA34 trend for higher probability trades.
Weekly filter reduces whipsaws during sideways periods while allowing trend participation.
Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.
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
    
    # Calculate weekly EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    typical_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    R1 = typical_1d + 1.1 * range_1d / 12
    S1 = typical_1d - 1.1 * range_1d / 12
    R2 = typical_1d + 1.1 * range_1d / 6
    S2 = typical_1d - 1.1 * range_1d / 6
    
    # Align Camarilla levels to 6h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for weekly EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with weekly uptrend
            if (close[i] > R1_aligned[i] and close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with weekly downtrend
            elif (close[i] < S1_aligned[i] and close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below S1 or weekly trend turns down
            if (close[i] < S1_aligned[i] or close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1 or weekly trend turns up
            if (close[i] > R1_aligned[i] or close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0