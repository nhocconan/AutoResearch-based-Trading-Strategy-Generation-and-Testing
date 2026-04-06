#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter + Volume Confirmation
Hypothesis: Ichimoku cloud from 1d timeframe provides strong trend direction, while 6m TK cross gives timely entries. Volume confirms breakout strength. Designed for 50-150 total trades over 4 years to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkouA, senkouB, chikou"""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    if n >= 9:
        for i in range(8, n):
            tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    if n >= 26:
        for i in range(25, n):
            kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    if n >= 26 and not np.all(np.isnan(tenkan)) and not np.all(np.isnan(kijun)):
        for i in range(n):
            if i + 26 < n and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                senkou_a[i + 26] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    if n >= 52:
        for i in range(51, n):
            senkou_b[i + 26] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    if n >= 26:
        for i in range(26, n):
            chikou[i] = close[i - 26]
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for Ichimoku
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6m timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # 6m TK cross (Tenkan/Kijun crossover)
    tk_cross = np.zeros(n, dtype=int)  # 1: bullish cross, -1: bearish cross, 0: no cross
    tenkan_6m = np.full(n, np.nan)
    kijun_6m = np.full(n, np.nan)
    
    # Calculate 6m Tenkan and Kijun for TK cross
    if n >= 9:
        for i in range(8, n):
            tenkan_6m[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    if n >= 26:
        for i in range(25, n):
            kijun_6m[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Detect TK cross (current vs previous)
    for i in range(1, n):
        if not np.isnan(tenkan_6m[i]) and not np.isnan(kijun_6m[i]) and \
           not np.isnan(tenkan_6m[i-1]) and not np.isnan(kijun_6m[i-1]):
            if tenkan_6m[i-1] <= kijun_6m[i-1] and tenkan_6m[i] > kijun_6m[i]:
                tk_cross[i] = 1  # Bullish cross
            elif tenkan_6m[i-1] >= kijun_6m[i-1] and tenkan_6m[i] < kijun_6m[i]:
                tk_cross[i] = -1  # Bearish cross
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > vol_ma * 1.5
    
    # Determine cloud direction from 1d Ichimoku
    # Cloud is bullish when price is above cloud, bearish when below
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from sufficient warmup
    start = max(60, 26)  # Need Ichimoku to develop
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or \
           np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: TK cross in opposite direction or price crosses cloud opposite
        if position == 1:  # long position
            # Exit: bearish TK cross or price drops below cloud
            if tk_cross[i] == -1 or price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bullish TK cross or price rises above cloud
            if tk_cross[i] == 1 or price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross in direction of cloud + volume
            if tk_cross[i] == 1 and price_above_cloud[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif tk_cross[i] == -1 and price_below_cloud[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals