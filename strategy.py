#!/usr/bin/env python3
"""
1d_Aroon_Trend_1wTrend_Filter
Hypothesis: Use Aroon indicator on daily timeframe to detect strong trends.
Combine with weekly trend filter (EMA34 on weekly) to ensure alignment.
Long when Aroon Up > 70 and Aroon Down < 30 with weekly uptrend.
Short when Aroon Down > 70 and Aroon Up < 30 with weekly downtrend.
Uses only daily and weekly timeframes to minimize noise and overtrading.
Target: 20-50 trades over 4 years (5-12.5/year) to reduce fee drag.
Works in both bull (catch strong uptrends) and bear (catch strong downtrends).
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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily Aroon (25 period)
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period - 1, n):
        # Periods since highest high
        highest_high_idx = i - np.argmax(high[i - aroon_period + 1:i + 1])
        periods_since_high = i - highest_high_idx
        aroon_up[i] = ((aroon_period - periods_since_high) / aroon_period) * 100
        
        # Periods since lowest low
        lowest_low_idx = i - np.argmin(low[i - aroon_period + 1:i + 1])
        periods_since_low = i - lowest_low_idx
        aroon_down[i] = ((aroon_period - periods_since_low) / aroon_period) * 100
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = aroon_period - 1  # Need Aroon calculation
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Aroon shows strong uptrend + weekly uptrend
            if aroon_up[i] > 70 and aroon_down[i] < 30 and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Aroon shows strong downtrend + weekly downtrend
            elif aroon_down[i] > 70 and aroon_up[i] < 30 and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Trend weakening or weekly trend reversal
            if aroon_up[i] < 50 or aroon_down[i] > 50 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Trend weakening or weekly trend reversal
            if aroon_down[i] < 50 or aroon_up[i] > 50 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals