#!/usr/bin/env python3
"""
6h_KAMA_Trend_Filter_Volume_Spike
Hypothesis: On 6h timeframe, use Kaufman's Adaptive Moving Average (KAMA) for trend detection, filtered by volume spike and price position relative to KAMA. This strategy captures trend continuation with low frequency to avoid fee drag. Works in both bull and bear markets as KAMA adapts to market volatility, reducing whipsaw during sideways periods. Target: 50-150 total trades over 4 years.
"""

name = "6h_KAMA_Trend_Filter_Volume_Spike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend context (optional filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Simple trend: price vs close 20 periods ago on 12h
    trend_12h = np.zeros(len(close_12h), dtype=bool)
    for i in range(20, len(close_12h)):
        trend_12h[i] = close_12h[i] > close_12h[i-20]
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h.astype(float))
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # 6h data for KAMA and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10, 2, 30)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # This needs fixing
    
    # Proper KAMA calculation
    diff = np.diff(close, prepend=close[0])
    abs_diff = np.abs(diff)
    
    # Direction over 10 periods
    direction = np.abs(np.diff(close, k=10, prepend=close[:10]))
    
    # Volatility sum over 10 periods
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + abs_diff[i]
        if i >= 10:
            volatility[i] -= abs_diff[i-10]
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility != 0
    er[mask] = direction[mask] / volatility[mask]
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10) and volume MA (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or 
            np.isnan(trend_12h_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Volume filter: current 6h volume > 1.8x 1d 20-period MA
        volume_filter = volume[i] > vol_ma20_1d_aligned[i] * 1.8
        
        # Session filter: 00-24 UTC (trade all hours for 6h)
        # No session filter for 6h to capture global moves
        
        if position == 0:
            # Long: price above KAMA with volume spike and 12h trend up
            if price_above_kama and volume_filter and trend_12h_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume spike and 12h trend down
            elif price_below_kama and volume_filter and trend_12h_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or volume dries up
            if price_below_kama or volume[i] < vol_ma20_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or volume dries up
            if price_above_kama or volume[i] < vol_ma20_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals