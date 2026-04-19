#!/usr/bin/env python3
"""
4h_KAMA_Trend_Volume_Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) identifies trend direction with minimal lag.
Price above KAMA = uptrend, below = downtrend. Volume confirmation filters false signals.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
Works in bull/bear via adaptive trend following and volume filter.
"""

name = "4h_KAMA_Trend_Volume_Confirmation"
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
    
    # KAMA (Kaufman Adaptive Moving Average) on close prices
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[er_length:] = change[er_length-1:] / volatility[er_length-1:]
        er[:er_length] = 0  # Not enough data
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # 4h data for KAMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    kama_4h = calculate_kama(df_4h['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(kama_4h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume confirmation
            if close[i] > kama_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation
            elif close[i] < kama_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA
            if close[i] < kama_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA
            if close[i] > kama_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals