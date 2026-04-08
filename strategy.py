#!/usr/bin/env python3
"""
6h_1d_ichimoku_cloud_v1
Hypothesis: 6-hour strategy using Ichimoku cloud from daily timeframe for trend direction,
with TK cross from 6h for entry timing. Long when price above daily cloud and TK crosses up;
short when price below daily cloud and TK crosses down. Includes volume confirmation.
Works in bull markets via trend following and in bear markets via short signals.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    n = len(high)
    if n < 52:
        return (np.full(n, np.nan), np.full(n, np.nan), 
                np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.maximum.accumulate(high)
    period9_low = np.minimum.accumulate(low)
    # For proper rolling window, we need to use convolution or loop
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    for i in range(8, n):
        tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    for i in range(25, n):
        kijun[i] = (np.max(high[i-24:i+1]) + np.min(low[i-24:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(26, n):
        if not (np.isnan(tenkan[i-26]) or np.isnan(kijun[i-26])):
            senkou_a[i] = (tenkan[i-26] + kijun[i-26]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(51, n):
        if i >= 51:
            max_52 = np.max(high[i-51:i+1])
            min_52 = np.min(low[i-51:i+1])
            senkou_b[i+26 if i+26 < n else n-1] = (max_52 + min_52) / 2
    
    # Chikou Span (Lagging Span): close shifted -22 periods
    chikou = np.full(n, np.nan)
    for i in range(n-22):
        chikou[i] = close[i+22]
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Calculate Kumo (cloud) boundaries: Senkou Span A and B
    # The cloud is between Senkou A and Senkou B
    # We need to determine if price is above or below the cloud
    # For simplicity, we'll use the midpoint of the cloud as reference
    kumo_top = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    kumo_mid = (kumo_top + kumo_bottom) / 2
    
    # Align Ichimoku components to 6-hour timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    kumo_mid_aligned = align_htf_to_ltf(prices, df_1d, kumo_mid)
    
    # Calculate TK cross on 6-hour timeframe for entry timing
    tenkan_6h, kijun_6h, _, _, _ = calculate_ichimoku(high, low, close)
    
    # Volume confirmation: 50-period average
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        # Determine if price is above or below daily cloud
        price_above_cloud = price > kumo_top_aligned[i]
        price_below_cloud = price < kumo_bottom_aligned[i]
        
        # TK cross signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 1:  # Long
            # Exit: price crosses below cloud bottom or TK cross down
            if price_below_cloud or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above cloud top or TK cross up
            if price_above_cloud or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud and TK cross up with volume confirmation
            if price_above_cloud and tk_cross_up and vol_ratio > 1.5:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud and TK cross down with volume confirmation
            elif price_below_cloud and tk_cross_down and vol_ratio > 1.5:
                position = -1
                signals[i] = -0.25
    
    return signals