#!/usr/bin/env python3
"""
1d_Aroon_Trend_1wTrend_Filter
Hypothesis: Aroon indicator (25-period) identifies strong trends when Aroon Up > 70 and Aroon Down < 30 (or vice versa).
Weekly trend filter ensures alignment with higher timeframe momentum. Works in both bull markets (strong uptrends)
and bear markets (strong downtrends) by capturing sustained directional moves. Target: 20-50 total trades over 4 years (5-12/year).
"""

name = "1d_Aroon_Trend_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly close for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Aroon indicator (25-period)
    period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low in the last 'period' bars
        window_high = high[i - period + 1:i + 1]
        window_low = low[i - period + 1:i + 1]
        
        # Calculate periods since high/low
        high_idx = np.argmax(window_high)  # 0 to period-1
        low_idx = np.argmin(window_low)    # 0 to period-1
        
        periods_since_high = period - 1 - high_idx
        periods_since_low = period - 1 - low_idx
        
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = period - 1  # warmup for Aroon
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend: price above/below EMA20
        weekly_uptrend = close[i] > ema20_1w_aligned[i]
        weekly_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long: Aroon Up > 70 (strong uptrend) and Aroon Down < 30 (weak downtrend) + weekly uptrend
            if aroon_up[i] > 70 and aroon_down[i] < 30 and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Aroon Down > 70 (strong downtrend) and Aroon Up < 30 (weak uptrend) + weekly downtrend
            elif aroon_down[i] > 70 and aroon_up[i] < 30 and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Aroon Down > 50 or weekly trend turns down
            if aroon_down[i] > 50 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Aroon Up > 50 or weekly trend turns up
            if aroon_up[i] > 50 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals