#!/usr/bin/env python3
"""
Strategy: 6h_1d_ichimoku_cloud_trend_v1
Timeframe: 6h
Hypothesis: Ichimoku Cloud on daily timeframe provides robust trend direction and support/resistance.
- Long when price is above the Kumo (cloud), Tenkan-sen > Kijun-sen, and price > Kijun-sen
- Short when price is below the Kumo, Tenkan-sen < Kijun-sen, and price < Kijun-sen
- Exit when price crosses back into the cloud or Tenkan/Kijun cross reverses
- Uses daily Ichimoku for higher timeframe trend filter, reducing whipsaws in 6h timeframe
- Works in both bull and bear markets by following the higher timeframe trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.maximum.accumulate(high)
    period9_low = np.minimum.accumulate(low)
    # For proper rolling window, we need to compute correctly
    tenkan_sen = np.full_like(high, np.nan)
    kijun_sen = np.full_like(high, np.nan)
    
    for i in range(len(high)):
        if i >= 8:  # 9 periods
            tenkan_sen[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
        if i >= 25:  # 26 periods
            kijun_sen[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full_like(high, np.nan)
    for i in range(len(high)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + 26
            if idx < len(high):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i >= 51:  # 52 periods
            val = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
            idx = i + 26
            if idx < len(high):
                senkou_span_b[idx] = val
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou_span = np.full_like(high, np.nan)
    for i in range(26, len(high)):
        chikou_span[i-26] = close[i]
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need enough data for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for Ichimoku
        # Skip if data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        senkou_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        senkou_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        price = close[i]
        tenkan = tenkan_1d_aligned[i]
        kijun = kijun_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price falls below cloud OR Tenkan crosses below Kijun
            if price < senkou_bottom or tenkan < kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price rises above cloud OR Tenkan crosses above Kijun
            if price > senkou_top or tenkan > kijun:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud, Tenkan > Kijun, and price > Kijun
            if price > senkou_top and tenkan > kijun and price > kijun:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud, Tenkan < Kijun, and price < Kijun
            elif price < senkou_bottom and tenkan < kijun and price < kijun:
                position = -1
                signals[i] = -0.25
    
    return signals