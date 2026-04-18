#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter
Ichimoku Cloud system with TK cross and cloud filter:
- Long when TK line crosses above Kijun + price above cloud (from 1d Ichimoku)
- Short when TK line crosses below Kijun + price below cloud
- Exit when TK line crosses back opposite direction
- Uses 1d Ichimoku cloud for trend filter (Senkou Span A/B)
- Designed for 15-25 trades/year per symbol
Works in both bull (captures trends) and bear (short breakdowns) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components."""
    n = len(high)
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over tenkan period
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan-1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan+1:i+1]) + np.min(low[i-tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over kijun period
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun-1, n):
        kijun_sen[i] = (np.max(high[i-kijun+1:i+1]) + np.min(low[i-kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted kijun periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(kijun-1, n):
        idx = i + kijun
        if idx < n:
            senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over senkou period shifted kijun
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou-1, n):
        idx = i + kijun
        if idx < n:
            senkou_span_b[idx] = (np.max(high[i-senkou+1:i+1]) + np.min(low[i-senkou+1:i+1])) / 2
    
    # Chikou Span (Lagging Span): close shifted -kijun periods (not used for signals)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Ichimoku
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_1d_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Ichimoku for TK cross
    tenkan_6h, kijun_6h, _, _ = calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need sufficient data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(tenkan_1d_6h[i]) or np.isnan(kijun_1d_6h[i]) or
            np.isnan(senkou_a_1d_6h[i]) or np.isnan(senkou_b_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a_1d_6h[i], senkou_b_1d_6h[i])
        cloud_bottom = min(senkou_a_1d_6h[i], senkou_b_1d_6h[i])
        
        # Check TK cross on 6h chart
        tk_cross_above = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_below = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: TK cross above + price above cloud
            if tk_cross_above and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud
            elif tk_cross_below and price_below_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross below (regardless of cloud)
            if tk_cross_below:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross above (regardless of cloud)
            if tk_cross_above:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0