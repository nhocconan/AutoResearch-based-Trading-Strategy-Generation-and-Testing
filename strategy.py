#!/usr/bin/env python3
"""
6h_ichimoku_cloud_v1
Hypothesis: Use weekly Ichimoku cloud (from 1w) as trend filter and 6h Tenkan/Kijun cross for entry.
- Only take long when price is above weekly cloud (bullish bias) and Tenkan crosses above Kijun on 6h
- Only take short when price is below weekly cloud (bearish bias) and Tenkan crosses below Kijun on 6h
- Exit on opposite Tenkan/Kijun cross or when price crosses back into weekly cloud
- Ichimoku components calculated on weekly timeframe for trend, entries on 6h for timing
- Designed to work in both bull and bear markets via cloud filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = close.copy()
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for Ichimoku (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku on weekly data
    tenkan_w, kijun_w, senkou_a_w, senkou_b_w, chikou_w = calculate_ichimoku(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    
    # Calculate weekly cloud boundaries (Senkou Span A/B)
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_w = np.maximum(senkou_a_w, senkou_b_w)
    cloud_bottom_w = np.minimum(senkou_a_w, senkou_b_w)
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_w)
    kijun_w_aligned = align_htf_to_ltf(prices, df_1w, kijun_w)
    cloud_top_w_aligned = align_htf_to_ltf(prices, df_1w, cloud_top_w)
    cloud_bottom_w_aligned = align_htf_to_ltf(prices, df_1w, cloud_bottom_w)
    
    # Calculate Ichimoku on 6h data for entry signals
    tenkan_6h, kijun_6h, _, _, _ = calculate_ichimoku(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(tenkan_w_aligned[i]) or np.isnan(kijun_w_aligned[i]) or
            np.isnan(cloud_top_w_aligned[i]) or np.isnan(cloud_bottom_w_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Determine if price is above/below weekly cloud
        price_above_cloud = close[i] > cloud_top_w_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_w_aligned[i]
        
        # Tenkan/Kijun cross signals on 6h
        tk_cross_above = (tenkan_6h[i-1] <= kijun_6h[i-1]) and (tenkan_6h[i] > kijun_6h[i])
        tk_cross_below = (tenkan_6h[i-1] >= kijun_6h[i-1]) and (tenkan_6h[i] < kijun_6h[i])
        
        if position == 1:  # Long
            # Exit: Tenkan crosses below Kijun OR price drops below weekly cloud
            if tk_cross_below or not price_above_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: Tenkan crosses above Kijun OR price rises above weekly cloud
            if tk_cross_above or not price_below_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: Price above weekly cloud AND Tenkan crosses above Kijun
            if price_above_cloud and tk_cross_above:
                position = 1
                signals[i] = 0.25
            # Short: Price below weekly cloud AND Tenkan crosses below Kijun
            elif price_below_cloud and tk_cross_below:
                position = -1
                signals[i] = -0.25
    
    return signals