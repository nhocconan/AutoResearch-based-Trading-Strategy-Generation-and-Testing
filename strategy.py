#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d Tenkan/Kijun cross and 1w Kijun filter.
- Long when: price > cloud, Tenkan > Kijun (1d), price > Kumo top (1d), and 1w Kijun rising
- Short when: price < cloud, Tenkan < Kijun (1d), price < Kumo bottom (1d), and 1w Kijun falling
- Uses Kumo twist (Senkou A/B cross) as trend strength filter
- Designed for 50-150 trades over 4 years to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components."""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    chikou = np.full(n, np.nan)
    
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
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < n:
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    for i in range(51, n):
        high_52 = np.max(high[i-51:i+1])
        low_52 = np.min(low[i-51:i+1])
        senkou_b[i + 26] = (high_52 + low_52) / 2 if (i + 26) < n else np.nan
    
    # Chikou Span (Lagging Span): close plotted 26 periods back
    for i in range(26, n):
        chikou[i - 26] = close[i]
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for Kijun filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku on 1d
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Calculate Kijun on 1w for trend filter
    _, kijun_1w, _, _, _ = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Align to 6h timeframe
    tenkan_1d_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_6h = align_htf_to_ltf(prices, df_1d, chikou_1d)
    kijun_1w_6h = align_htf_to_ltf(prices, df_1w, kijun_1w)
    
    # Determine cloud top and bottom
    kumo_top_1d_6h = np.maximum(senkou_a_1d_6h, senkou_b_1d_6h)
    kumo_bottom_1d_6h = np.minimum(senkou_a_1d_6h, senkou_b_1d_6h)
    
    # Kumo twist (Senkou A/B cross) - trend strength
    kumo_twist = senkou_a_1d_6h - senkou_b_1d_6h
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need Ichimoku calculation (max 52+26)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_1d_6h[i]) or np.isnan(kijun_1d_6h[i]) or 
            np.isnan(kumo_top_1d_6h[i]) or np.isnan(kumo_bottom_1d_6h[i]) or
            np.isnan(kijun_1w_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Kijun trend: rising if current > previous 6 periods ago
        kijun_1w_rising = not np.isnan(kijun_1w_6h[i]) and i >= 6 and kijun_1w_6h[i] > kijun_1w_6h[i-6]
        kijun_1w_falling = not np.isnan(kijun_1w_6h[i]) and i >= 6 and kijun_1w_6h[i] < kijun_1w_6h[i-6]
        
        if position == 0:
            # Long: price > cloud, Tenkan > Kijun, Kumo twist bullish, Kijun rising, volume
            if (close[i] > kumo_top_1d_6h[i] and 
                tenkan_1d_6h[i] > kijun_1d_6h[i] and 
                kumo_twist[i] > 0 and 
                kijun_1w_rising and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price < cloud, Tenkan < Kijun, Kumo twist bearish, Kijun falling, volume
            elif (close[i] < kumo_bottom_1d_6h[i] and 
                  tenkan_1d_6h[i] < kijun_1d_6h[i] and 
                  kumo_twist[i] < 0 and 
                  kijun_1w_falling and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < cloud bottom or Tenkan < Kijun
            if close[i] < kumo_bottom_1d_6h[i] or tenkan_1d_6h[i] < kijun_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > cloud top or Tenkan > Kijun
            if close[i] > kumo_top_1d_6h[i] or tenkan_1d_6h[i] > kijun_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_1dTK_1wKijun"
timeframe = "6h"
leverage = 1.0