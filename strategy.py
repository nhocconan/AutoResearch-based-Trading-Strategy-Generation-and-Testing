#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_1dCloud_Filter
# Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with 1d cloud filter (price above/below 1d Kumo).
# Uses Senkou Span A/B from daily timeframe to define bull/bear regime.
# Long when TK cross bullish and price > 1d Senkou Span A (cloud top).
# Short when TK cross bearish and price < 1d Senkou Span B (cloud bottom).
# Designed for low-frequency, high-conviction trades in both bull and bear markets.

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter"
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
    
    # Get daily data for cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_6h = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_6h[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + low)/2
    kijun_6h = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun_6h[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, shifted 26 periods ahead
    senkou_span_a_6h = np.full(n, np.nan)
    for i in range(n):
        idx = i + kijun_period  # Shift forward by 26 periods
        if idx < n and not np.isnan(tenkan_6h[i]) and not np.isnan(kijun_6h[i]):
            senkou_span_a_6h[idx] = (tenkan_6h[i] + kijun_6h[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + low)/2, shifted 26 periods ahead
    senkou_span_b_6h = np.full(n, np.nan)
    for i in range(n):
        idx = i + kijun_period  # Shift forward by 26 periods
        if idx < n and i >= senkou_span_b_period - 1:
            senkou_span_b_6h[idx] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Calculate 1d Ichimoku cloud components
    # Tenkan-sen (9-period) on daily
    tenkan_1d = np.full_like(close_1d, np.nan)
    for i in range(tenkan_period - 1, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (26-period) on daily
    kijun_1d = np.full_like(close_1d, np.nan)
    for i in range(kijun_period - 1, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (1d): (Tenkan + Kijun)/2
    senkou_span_a_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_span_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Senkou Span B (1d): (52-period high + low)/2
    senkou_span_b_1d = np.full_like(close_1d, np.nan)
    for i in range(senkou_span_b_period - 1, len(close_1d)):
        senkou_span_b_1d[i] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # Align 1d cloud to 6h timeframe (wait for daily close)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are ready
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + kijun_period  # 26+26=52
    
    for i in range(start_idx, n):
        # Skip if any required data is not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine TK cross
        tk_bullish = tenkan_6h[i] > kijun_6h[i]
        tk_bearish = tenkan_6h[i] < kijun_6h[i]
        
        # Determine price vs 1d cloud
        price_above_cloud_top = close[i] > senkou_span_a_1d_aligned[i]
        price_below_cloud_bottom = close[i] < senkou_span_b_1d_aligned[i]
        
        if position == 0:
            # Enter long: bullish TK cross AND price above 1d cloud top (Senkou Span A)
            if tk_bullish and price_above_cloud_top:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish TK cross AND price below 1d cloud bottom (Senkou Span B)
            elif tk_bearish and price_below_cloud_bottom:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish TK cross OR price falls below cloud bottom
            if tk_bearish or close[i] < senkou_span_b_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud top
            if tk_bullish or close[i] > senkou_span_a_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals