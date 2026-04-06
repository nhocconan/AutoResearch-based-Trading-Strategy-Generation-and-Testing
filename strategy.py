#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK cross signals momentum shifts, filtered by 1d trend (price above/below 1d Kumo) to align with higher timeframe bias. Works in bull (buy when price above cloud + TK cross up) and bear (sell when price below cloud + TK cross down). Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = np.full(n, np.nan)
    for i in range(8, n):
        tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = np.full(n, np.nan)
    for i in range(25, n):
        kijun[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(26, n):
        if not np.isnan(tenkan[i-26]) and not np.isnan(kijun[i-26]):
            senkou_a[i] = (tenkan[i-26] + kijun[i-26]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(51, n):
        if i - 26 >= 0:
            high_52 = np.max(high[i-51:i+1]) if i-51 >= 0 else np.max(high[:i+1])
            low_52 = np.min(low[i-51:i+1]) if i-51 >= 0 else np.min(low[:i+1])
            senkou_b[i] = (high_52 + low_52) / 2
    
    # Get 1d data for trend filter (price vs 1d Kumo)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku components
    tenkan_1d = np.full(len(close_1d), np.nan)
    kijun_1d = np.full(len(close_1d), np.nan)
    senkou_a_1d = np.full(len(close_1d), np.nan)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    
    for i in range(8, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    for i in range(25, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    for i in range(26, len(close_1d)):
        if not np.isnan(tenkan_1d[i-26]) and not np.isnan(kijun_1d[i-26]):
            senkou_a_1d[i] = (tenkan_1d[i-26] + kijun_1d[i-26]) / 2
    for i in range(51, len(close_1d)):
        high_52 = np.max(high_1d[i-51:i+1]) if i-51 >= 0 else np.max(high_1d[:i+1])
        low_52 = np.min(low_1d[i-51:i+1]) if i-51 >= 0 else np.min(low_1d[:i+1])
        senkou_b_1d[i] = (high_52 + low_52) / 2
    
    # 1d Kumo (cloud) boundaries: Senkou Span A and B
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Kumo to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # TK cross signals
    tk_cross = np.zeros(n)  # 1 for bullish cross, -1 for bearish cross
    for i in range(1, n):
        if not np.isnan(tenkan[i-1]) and not np.isnan(kijun[i-1]) and \
           not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            if tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]:
                tk_cross[i] = 1  # bullish cross
            elif tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]:
                tk_cross[i] = -1  # bearish cross
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period (need enough data for Ichimoku)
    start = 60
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Determine cloud color and position
        # Cloud is green (bullish) when Senkou A > Senkou B
        # Cloud is red (bearish) when Senkou A < Senkou B
        cloud_green = senkou_a[i] > senkou_b[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > senkou_a[i] and close[i] > senkou_b[i]
        price_below_cloud = close[i] < senkou_a[i] and close[i] < senkou_b[i]
        price_in_cloud = not (price_above_cloud or price_below_cloud)
        
        # 1d trend filter: price relative to 1d Kumo
        price_above_1d_kumo = close[i] > kumo_top_aligned[i]
        price_below_1d_kumo = close[i] < kumo_bottom_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Kumo OR TK cross bearish
            if price_below_cloud or tk_cross[i] == -1:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price closes above Kumo OR TK cross bullish
            if price_above_cloud or tk_cross[i] == 1:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum bars between trades
            if bars_since_exit >= 12:
                # Long: price above cloud + bullish TK cross + price above 1d Kumo
                if price_above_cloud and tk_cross[i] == 1 and price_above_1d_kumo:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: price below cloud + bearish TK cross + price below 1d Kumo
                elif price_below_cloud and tk_cross[i] == -1 and price_below_1d_kumo:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals