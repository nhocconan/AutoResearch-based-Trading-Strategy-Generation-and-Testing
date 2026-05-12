#!/usr/bin/env python3
# 1d_KAMA_Trend_1wTrend
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 1d to capture trend direction,
# filtered by 1w KAMA trend for higher timeframe confirmation. Enter long when 1d KAMA > 1w KAMA,
# enter short when 1d KAMA < 1w KAMA. Exit on opposite crossover.
# Designed for low frequency (10-30 trades/year) to avoid fee drag. Works in bull (catch trends)
# and bear (catch downtrends) with dual timeframe alignment.

name = "1d_KAMA_Trend_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, period=10, fast=2, slow=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Returns KAMA array.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    if n < period:
        return kama
    
    # Efficiency ratio
    change = np.abs(np.diff(close, n=period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    
    # Handle first period elements
    er = np.zeros(n)
    er[period:] = change / np.where(volatility[period:] == 0, 1, volatility[period:])
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on daily data
    kama_1d = kama(close, period=10, fast=2, slow=30)
    
    # Calculate KAMA on weekly data
    kama_1w = kama(close_1w, period=10, fast=2, slow=30)
    
    # Align weekly KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure KAMA is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_1d[i]) or np.isnan(kama_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: daily KAMA above/below weekly KAMA
        kama_up = kama_1d[i] > kama_1w_aligned[i]
        kama_down = kama_1d[i] < kama_1w_aligned[i]
        
        if position == 0:
            # LONG: 1d KAMA > 1w KAMA
            if kama_up:
                signals[i] = 0.25
                position = 1
            # SHORT: 1d KAMA < 1w KAMA
            elif kama_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: 1d KAMA < 1w KAMA (opposite crossover)
            if kama_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 1d KAMA > 1w KAMA (opposite crossover)
            if kama_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals