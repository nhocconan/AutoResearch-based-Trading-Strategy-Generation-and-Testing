#!/usr/bin/env python3
"""
6h_1D_Ichimoku_TK_Cross_CloudFilter
Hypothesis: On 6h timeframe, use daily Ichimoku cloud for trend filtering (price above/below cloud) and Tenkan/Kijun cross for entry.
Works in both bull and bear markets by only trading in direction of daily trend (cloud), avoiding counter-trend whipsaws.
Targets 15-25 trades/year to minimize fee drift while capturing strong trending moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.maximum.accumulate(high)
    period9_low = np.minimum.accumulate(low)
    # For proper rolling window, we need to use convolution-like approach
    tenkan = np.full_like(high, np.nan)
    kijun = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i >= 8:  # 9 periods
            tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
        if i >= 25:  # 26 periods
            kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_a = np.full_like(high, np.nan)
    for i in range(len(tenkan)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            if i + 26 < len(senkou_a):
                senkou_a[i + 26] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 periods ahead
    senkou_b = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= 51:  # 52 periods
            senkou_b[i + 26] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily Ichimoku
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need Ichimoku (52 periods for Senkou B)
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top_aligned[i]) or
            np.isnan(cloud_bottom_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine trend from daily cloud: price above cloud = uptrend, below = downtrend
        above_cloud = price > cloud_top_aligned[i]
        below_cloud = price < cloud_bottom_aligned[i]
        
        # TK cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 0:
            # Long: price above cloud AND TK cross up
            if above_cloud and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND TK cross down
            elif below_cloud and tk_cross_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below cloud OR TK cross down
            if not above_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above cloud OR TK cross up
            if not below_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_1D_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0