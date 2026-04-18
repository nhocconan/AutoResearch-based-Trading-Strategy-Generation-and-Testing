#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Breakout_Volume_Trend
Hypothesis: 1-hour breakouts above R1 or below S1 of daily Camarilla pivots with volume confirmation and 4-hour EMA trend filter.
Uses 4h EMA for trend direction (less noisy than 1d) and 1h for precise entry timing.
Targets 20-40 trades/year by requiring volume spike and strong trend alignment.
Designed to work in both bull and bear markets via trend filter and volatility-based entry.
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
    
    # Calculate daily Camarilla pivot levels (from 1d data)
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
    
    # 4-hour EMA34 trend filter (balanced responsiveness and smoothness)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = np.full(len(close_4h), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_4h)):
        if i == 34:
            ema34_4h[i] = np.mean(close_4h[0:35])
        else:
            ema34_4h[i] = close_4h[i] * k + ema34_4h[i-1] * (1 - k)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume spike: current volume > 2.0 x 20-period average (1h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (reduces noise outside active trading hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume spike and 4h uptrend
            if (close[i] > r1_1d_aligned[i] and vol_spike[i] and 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: break below S1 with volume spike and 4h downtrend
            elif (close[i] < s1_1d_aligned[i] and vol_spike[i] and 
                  close[i] < ema34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: close below pivot or 4h trend turns down
            if (close[i] < pivot_1d_aligned[i] or close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: close above pivot or 4h trend turns up
            if (close[i] > pivot_1d_aligned[i] or close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Camarilla_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0