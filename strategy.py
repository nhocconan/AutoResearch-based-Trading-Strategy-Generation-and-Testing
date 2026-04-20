#!/usr/bin/env python3
# 6h_Ichimoku_TenkanKijun_Cross_With_1D_Trend_Filter
# Hypothesis: Combines 6h Ichimoku Tenkan/Kijun cross signals with 1d Ichimoku cloud for trend filtering.
# In bull markets (price above 1d cloud): long when Tenkan crosses above Kijun.
# In bear markets (price below 1d cloud): short when Tenkan crosses below Kijun.
# Requires cloud thickness > 0 to avoid weak signals in ranging markets.
# Uses standard Ichimoku formulas: Tenkan=(9-period high+low)/2, Kijun=(26-period high+low)/2.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Ichimoku_TenkanKijun_Cross_With_1D_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    tenkan_1d = np.full_like(high_1d, np.nan)
    for i in range(8, len(high_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    kijun_1d = np.full_like(high_1d, np.nan)
    for i in range(25, len(high_1d)):
        kijun_1d[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    senkou_span_a_1d = np.full_like(high_1d, np.nan)
    for i in range(25, len(high_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_span_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Senkou Span B (Leading Span B): 52-period high-low midpoint, plotted 26 periods ahead
    senkou_span_b_1d = np.full_like(high_1d, np.nan)
    for i in range(51, len(high_1d)):
        senkou_span_b_1d[i] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Calculate 6h Ichimoku for entry signals
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    tenkan_6h = np.full_like(high, np.nan)
    for i in range(8, len(high)):
        tenkan_6h[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    kijun_6h = np.full_like(high, np.nan)
    for i in range(25, len(high)):
        kijun_6h[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 50)  # Need enough data for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d Ichimoku cloud boundaries
        cloud_top = max(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        cloud_bottom = min(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        cloud_thickness = cloud_top - cloud_bottom
        
        # Skip if cloud is too thin (ranging market)
        if cloud_thickness < 0.001 * close[i]:  # Less than 0.1% of price
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d Ichimoku cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        if position == 0:
            # Long: price above cloud + Tenkan crosses above Kijun
            if price_above_cloud and tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + Tenkan crosses below Kijun
            elif price_below_cloud and tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price falls below cloud or Tenkan crosses below Kijun
            if close[i] < cloud_bottom or (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price rises above cloud or Tenkan crosses above Kijun
            if close[i] > cloud_top or (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals