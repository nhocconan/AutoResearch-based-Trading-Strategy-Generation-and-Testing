#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filtered_Trend
Hypothesis: Use 1d Ichimoku cloud as primary trend filter and TK cross for entry timing on 6h timeframe.
The Ichimoku cloud (Senkou Span A/B) from 1d acts as a strong support/resistance zone and trend filter.
Tenkan/Kijun cross provides momentum signals aligned with the higher timeframe trend.
This combination should work in both bull and bear markets by only taking trades in the direction of the 1d cloud trend.
Target: 15-30 trades/year (60-120 total) to minimize fee drag.
"""

name = "6h_Ichimoku_Cloud_Filtered_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A, Senkou B"""
    n1 = len(high)
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = np.maximum.accumulate(high)
    period9_low = np.minimum.accumulate(low)
    # For proper windowed calculation, we need rolling max/min
    tenkan = np.full(n1, np.nan)
    kijun = np.full(n1, np.nan)
    
    # Calculate rolling max/min for 9 and 26 periods
    for i in range(n1):
        if i >= 8:  # 9 periods
            start9 = i - 8
            tenkan[i] = (np.max(high[start9:i+1]) + np.min(low[start9:i+1])) / 2
        if i >= 25:  # 26 periods
            start26 = i - 25
            kijun[i] = (np.max(high[start26:i+1]) + np.min(low[start26:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = np.full(n1, np.nan)
    valid_mask = ~(np.isnan(tenkan) | np.isnan(kijun))
    senkou_a[valid_mask] = (tenkan[valid_mask] + kijun[valid_mask]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b = np.full(n1, np.nan)
    for i in range(n1):
        if i >= 51:  # 52 periods
            start52 = i - 51
            senkou_b[i] = (np.max(high[start52:i+1]) + np.min(low[start52:i+1])) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on 1d data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for Ichimoku calculation (52 periods for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or \
           np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend based on cloud relationship
        # Bullish: price above cloud, Bearish: price below cloud
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i] and \
                      tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i] and \
                        tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1]
        
        if position == 0:
            # Long: price above cloud + TK cross up
            if price_above_cloud and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down
            elif price_below_cloud and tk_cross_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below cloud or TK cross down
            if not price_above_cloud or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud or TK cross up
            if not price_below_cloud or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals