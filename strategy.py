#!/usr/bin/env python3
"""
6h_ichimoku_cloud_v1
Hypothesis: Ichimoku system on 6h with 1d trend filter.
- Use 1d Kumo (cloud) to filter long/short bias: price above cloud = long bias, below cloud = short bias
- Entry: 6h Tenkan/Kijun cross in direction of 1d cloud bias, with price outside cloud
- Exit: Opposite TK cross or price re-enters cloud
- Works in bull/bear via cloud filter; TK cross catches momentum within trend
- Target: 20-40 trades/year
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, senkou_a, senkou_b, kijun"""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    for i in range(8, n):
        high_9 = np.max(high[i-8:i+1])
        low_9 = np.min(low[i-8:i+1])
        tenkan[i] = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    for i in range(25, n):
        high_26 = np.max(high[i-25:i+1])
        low_26 = np.min(low[i-25:i+1])
        kijun[i] = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < n:
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    for i in range(51, n):
        high_52 = np.max(high[i-51:i+1])
        low_52 = np.min(low[i-51:i+1])
        senkou_b[i + 26] = (high_52 + low_52) / 2 if (i + 26) < n else np.nan
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 6h Ichimoku
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # Get 1d data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d cloud to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d cloud bias
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross signals
        tk_cross_bull = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_bear = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 1:  # Long
            # Exit: bearish TK cross or price re-enters cloud
            if tk_cross_bear or (close[i] >= cloud_bottom and close[i] <= cloud_top):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: bullish TK cross or price re-enters cloud
            if tk_cross_bull or (close[i] >= cloud_bottom and close[i] <= cloud_top):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish TK cross + price above 1d cloud
            if tk_cross_bull and price_above_cloud:
                position = 1
                signals[i] = 0.25
            # Enter short: bearish TK cross + price below 1d cloud
            elif tk_cross_bear and price_below_cloud:
                position = -1
                signals[i] = -0.25
    
    return signals