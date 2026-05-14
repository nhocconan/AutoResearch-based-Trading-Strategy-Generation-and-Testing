#!/usr/bin/env python3
"""
6h_Ichimoku_1dCloud_Filter_V1
Hypothesis: Ichimoku Cloud from 1d timeframe acts as major support/resistance. Price above/below cloud with Tenkan-Kijun cross on 6h provides high-probability trend continuation. Cloud filter reduces whipsaw in sideways markets. Works in bull/bear by only taking trades aligned with higher timeframe cloud color.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, senkouA, senkouB"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.maximum.accumulate(high)
    period9_low = np.minimum.accumulate(low)
    # For simplicity, using rolling max/min - in practice would need proper window
    tenkan = np.full_like(high, np.nan)
    for i in range(len(high)):
        start = max(0, i - 8)
        tenkan[i] = (np.max(high[start:i+1]) + np.min(low[start:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full_like(high, np.nan)
    for i in range(len(high)):
        start = max(0, i - 25)
        kijun[i] = (np.max(high[start:i+1]) + np.min(low[start:i+1])) / 2
    
    senkou_a = np.full_like(high, np.nan)
    senkou_b = np.full_like(high, np.nan)
    for i in range(len(high)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    for i in range(len(high)):
        start = max(0, i - 51)
        senkou_b[i] = (np.max(high[start:i+1]) + np.min(low[start:i+1])) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    # Load 1d data once for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    cloud_color_1d = senkou_a_1d - senkou_b_1d  # >0 = bullish cloud, <0 = bearish cloud
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    cloud_color_aligned = align_htf_to_ltf(prices, df_1d, cloud_color_1d)
    
    # 6h Tenkan-Kijun cross for entry signal
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    tenkan_6h = np.full_like(close_6h, np.nan)
    kijun_6h = np.full_like(close_6h, np.nan)
    
    for i in range(len(close_6h)):
        start = max(0, i - 8)
        tenkan_6h[i] = (np.max(high_6h[start:i+1]) + np.min(low_6h[start:i+1])) / 2
        start = max(0, i - 25)
        kijun_6h[i] = (np.max(high_6h[start:i+1]) + np.min(low_6h[start:i+1])) / 2
    
    # TK cross: 1 = bullish cross (tenkan > kijun), -1 = bearish cross (tenkan < kijun)
    tk_cross = np.zeros_like(close_6h)
    for i in range(1, len(close_6h)):
        if not np.isnan(tenkan_6h[i]) and not np.isnan(kijun_6h[i]):
            if tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]:
                tk_cross[i] = 1  # bullish cross
            elif tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]:
                tk_cross[i] = -1  # bearish cross
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if NaN in critical values
        if np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or np.isnan(cloud_color_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        cloud_top = cloud_top_aligned[i]
        cloud_bottom = cloud_bottom_aligned[i]
        cloud_color = cloud_color_aligned[i]
        tk_signal = tk_cross[i]
        
        # Cloud filter: only trade when price is clearly above/below cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        if position == 0:
            # Long: bullish TK cross + price above cloud + bullish cloud
            if tk_signal == 1 and price_above_cloud and cloud_color > 0:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross + price below cloud + bearish cloud
            elif tk_signal == -1 and price_below_cloud and cloud_color < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below cloud or bearish TK cross
            if price < cloud_bottom or tk_signal == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or bullish TK cross
            if price > cloud_top or tk_signal == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dCloud_Filter_V1"
timeframe = "6h"
leverage = 1.0