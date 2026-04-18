#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrendFilter_v1
Hypothesis: 1-hour breakouts above R1 or below S1 of daily Camarilla pivots with 4-hour EMA34 trend filter.
Trades only during 08-20 UTC to avoid low-volume sessions. Uses 4h trend direction to avoid counter-trend trades.
Target: 60-150 total trades over 4 years (15-37/year) with controlled risk via position size 0.20.
Designed to work in both bull and bear markets by following higher timeframe trend.
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
    
    # Calculate 1-day Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R1, S1)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + range_1d * 1.1 / 12
    s1_1d = pivot_1d - range_1d * 1.1 / 12
    
    # Align 1-day levels to 1h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4-hour EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = np.full(len(close_4h), np.nan)
    for i in range(34, len(close_4h)):
        if i == 34:
            ema34_4h[i] = np.mean(close_4h[0:35])
        else:
            k = 2 / (34 + 1)
            ema34_4h[i] = close_4h[i] * k + ema34_4h[i-1] * (1 - k)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 34)  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema34_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with 4h uptrend
            if close[i] > r1_1d_aligned[i] and close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with 4h downtrend
            elif close[i] < s1_1d_aligned[i] and close[i] < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: close below pivot or 4h trend turns down
            if close[i] < pivot_1d_aligned[i] or close[i] < ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: close above pivot or 4h trend turns up
            if close[i] > pivot_1d_aligned[i] or close[i] > ema34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrendFilter_v1"
timeframe = "1h"
leverage = 1.0