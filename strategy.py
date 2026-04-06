#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter
Hypothesis: Use 1d Ichimoku cloud as trend filter and 6h Tenkan/Kijun cross for entry timing.
Ichimoku provides dynamic support/resistance and trend direction. In bull markets, price stays above cloud;
in bear markets, price stays below cloud. TK cross catches momentum shifts within the trend.
Target: 80-150 total trades over 4 years (20-38/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components. Returns (tenkan, kijun, senkou_a, senkou_b, chikou)"""
    n = len(high)
    tenkan_sen = (np.maximum.accumulate(high)[:tenkan] + np.minimum.accumulate(low)[:tenkan]) / 2
    kijun_sen = (np.maximum.accumulate(high)[:kijun] + np.minimum.accumulate(low)[:kijun]) / 2
    
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    chikou = np.full(n, np.nan)
    
    for i in range(tenkan-1, n):
        tenkan[i] = (np.max(high[i-tenkan+1:i+1]) + np.min(low[i-tenkan+1:i+1])) / 2
    
    for i in range(kijun-1, n):
        kijun[i] = (np.max(high[i-kijun+1:i+1]) + np.min(low[i-kijun+1:i+1])) / 2
    
    for i in range(kijun-1, n):
        senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    for i in range(senkou-1, n):
        senkou_b[i] = (np.max(high[i-senkou+1:i+1]) + np.min(low[i-senkou+1:i+1])) / 2
    
    for i in range(n):
        if i + 26 < n:
            chikou[i] = close[i + 26]
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on 1d for trend filter
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need enough data for Ichimoku)
    start = max(52, 26)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from 1d Ichimoku
        # Bullish trend: price above cloud (price > Senkou A and Senkou B)
        # Bearish trend: price below cloud (price < Senkou A and Senkou B)
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        is_bullish_trend = close[i] > cloud_top_1d
        is_bearish_trend = close[i] < cloud_bottom_1d
        
        # TK cross signals on 6h
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Check exits
        if position == 1:  # long position
            # Exit: TK cross down or price falls below cloud
            if tk_cross_down or close[i] < cloud_bottom_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross up or price rises above cloud
            if tk_cross_up or close[i] > cloud_top_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries - only in direction of 1d trend
            # Long: bullish trend + TK cross up
            if is_bullish_trend and tk_cross_up:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: bearish trend + TK cross down
            elif is_bearish_trend and tk_cross_down:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals