#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_With_Volume_Filter
Long when Tenkan crosses above Kijun AND price is above Kumo (cloud) with volume > 1.5x average.
Short when Tenkan crosses below Kijun AND price is below Kumo with volume > 1.5x average.
Exit when Tenkan crosses back through Kijun.
Uses Ichimoku cloud as dynamic support/resistance and trend filter.
Target: 60-120 trades over 4 years (15-30/year).
"""

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
    volume = prices['volume'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    kumo_shift = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(high[i - tenkan_period + 1:i + 1]) + np.min(low[i - tenkan_period + 1:i + 1])) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(high[i - kijun_period + 1:i + 1]) + np.min(low[i - kijun_period + 1:i + 1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + kumo_shift
            if idx < n:
                senkou_span_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 ahead
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        val = (np.max(high[i - senkou_span_b_period + 1:i + 1]) + np.min(low[i - senkou_span_b_period + 1:i + 1])) / 2
        idx = i + kumo_shift
        if idx < n:
            senkou_span_b[idx] = val
    
    # Volume moving average for confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Tenkan, Kijun, Senkou spans, and volume MA
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period + kumo_shift, vol_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Determine Kumo (cloud) boundaries
        senkou_top = max(senkou_span_a[i], senkou_span_b[i])
        senkou_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above Kumo with volume filter
            if (tenkan[i] > kijun[i] and tenkan[i - 1] <= kijun[i - 1] and 
                price > senkou_top and vol_filter):
                signals[i] = size
                position = 1
            # Short: Tenkan crosses below Kijun AND price below Kumo with volume filter
            elif (tenkan[i] < kijun[i] and tenkan[i - 1] >= kijun[i - 1] and 
                  price < senkou_bottom and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun
            if tenkan[i] < kijun[i] and tenkan[i - 1] >= kijun[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun
            if tenkan[i] > kijun[i] and tenkan[i - 1] <= kijun[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0