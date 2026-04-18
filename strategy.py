#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_12h_Trend_Filter
Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
Using 12h KAMA for trend filter and 4h KAMA for entry signals reduces whipsaw.
Works in both bull and bear markets by following the dominant trend on higher timeframe.
Target: 20-40 trades/year to minimize fee drag.
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
    
    # 4h KAMA for entry signals
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_series - close_series.shift(10))
    volatility = abs(close_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 12h KAMA for trend filter (higher timeframe)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_series_12h = pd.Series(close_12h)
    change_12h = abs(close_series_12h - close_series_12h.shift(10))
    volatility_12h = abs(close_series_12h.diff()).rolling(window=10, min_periods=10).sum()
    er_12h = change_12h / volatility_12h.replace(0, 1e-10)
    sc_12h = (er_12h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_12h = np.zeros_like(close_12h)
    kama_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    # Align to 4h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(kama_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Go long when price above both 4h and 12h KAMA
            if close[i] > kama[i] and close[i] > kama_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Go short when price below both 4h and 12h KAMA
            elif close[i] < kama[i] and close[i] < kama_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price crosses below 4h KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above 4h KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Trend_With_12h_Trend_Filter"
timeframe = "4h"
leverage = 1.0