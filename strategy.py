#!/usr/bin/env python3
"""
6h_1w_Donchian20_Breakout_WeeklyTrend
Hypothesis: 6-hour Donchian(20) breakouts in direction of weekly trend (price above/below weekly SMA50) yield high-probability trades. Uses weekly trend filter to avoid counter-trend whipsaws, works in bull/bear markets by only taking breakouts aligned with weekly trend. Targets 15-35 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_sma(arr, period):
    """Simple Moving Average"""
    sma = np.full_like(arr, np.nan, dtype=float)
    for i in range(period - 1, len(arr)):
        sma[i] = np.mean(arr[i - period + 1:i + 1])
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = calculate_sma(close_1w, 50)
    
    # Align weekly SMA to 6h timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if weekly SMA not ready
        if np.isnan(sma_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 6h Donchian(20) channels (use previous 20 bars for breakout)
        high_20 = prices['high'].iloc[i-20:i].max()
        low_20 = prices['low'].iloc[i-20:i].min()
        
        price = prices['close'].iloc[i]
        weekly_sma = sma_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price above weekly SMA50 (uptrend)
            if price > high_20 and price > weekly_sma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price below weekly SMA50 (downtrend)
            elif price < low_20 and price < weekly_sma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or price crosses below weekly SMA50
            if price < low_20 or price < weekly_sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or price crosses above weekly SMA50
            if price > high_20 or price > weekly_sma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_Donchian20_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0