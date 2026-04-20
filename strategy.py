#!/usr/bin/env python3
# 1d_KAMA_Trend_With_1w_Trend_Filter
# Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) on 1d to determine trend direction,
# filtered by weekly trend (price > weekly KAMA). KAMA adapts to market noise, reducing whipsaw
# in ranging markets while capturing trends. Weekly filter ensures alignment with higher timeframe
# momentum, improving performance in both bull and bear markets. Target: 20-60 trades/year.

name = "1d_KAMA_Trend_With_1w_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    def kama(close, er_period=10, fast=2, slow=30):
        n = len(close)
        kama_arr = np.full(n, np.nan)
        if n < er_period:
            return kama_arr
        
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Handle first er_period elements
        er = np.full(n, np.nan)
        for i in range(er_period, n):
            if volatility[i] != 0:
                er[i] = change[i-er_period] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Initialize KAMA
        kama_arr[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            if not np.isnan(sc[i]):
                kama_arr[i] = kama_arr[i-1] + sc[i] * (close[i] - kama_arr[i-1])
            else:
                kama_arr[i] = kama_arr[i-1]
        return kama_arr
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    kama_1w = kama(close_1w, er_period=10, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Calculate daily KAMA
    kama_daily = kama(close, er_period=10, fast=2, slow=30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_daily[i]) or np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > daily KAMA AND price > weekly KAMA (uptrend on both)
            if close[i] > kama_daily[i] and close[i] > kama_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < daily KAMA AND price < weekly KAMA (downtrend on both)
            elif close[i] < kama_daily[i] and close[i] < kama_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens on either timeframe
            if close[i] < kama_daily[i] or close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend weakens on either timeframe
            if close[i] > kama_daily[i] or close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals