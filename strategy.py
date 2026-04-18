#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v2
Hypothesis: Use Ichimoku cloud from 1d timeframe as trend filter and 6h price action for entries.
In bull markets, price above cloud + TK cross up triggers long; in bear markets, price below cloud + TK cross down triggers short.
The cloud acts as dynamic support/resistance, reducing whipsaws in sideways markets. TK cross provides timely entry signals.
Designed for low trade frequency (target: 15-30/year) with strong performance in both bull and bear markets.
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.full(len(high_1d), np.nan)
    period9_low = np.full(len(low_1d), np.nan)
    for i in range(8, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-8:i+1])
        period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.full(len(high_1d), np.nan)
    period26_low = np.full(len(low_1d), np.nan)
    for i in range(25, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-25:i+1])
        period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.full(len(high_1d), np.nan)
    period52_low = np.full(len(low_1d), np.nan)
    for i in range(51, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-51:i+1])
        period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper look-ahead prevention)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Determine cloud boundaries (Senkou Span A and B)
    # The cloud is plotted 26 periods ahead, so we need to shift the spans back for current price comparison
    # However, since we're using align_htf_to_ltf which already handles the timing correctly,
    # we use the values as-is for current cloud assessment
    upper_cloud = np.maximum(senkou_a_6h, senkou_b_6h)
    lower_cloud = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK Cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Crossed above (current above, previous below or equal)
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Crossed below (current below, previous above or equal)
    
    # Fix the TK cross logic - need to compare with previous values
    tk_cross_above = np.zeros(n, dtype=bool)
    tk_cross_below = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(tenkan_6h[i-1]) or np.isnan(kijun_6h[i-1])):
            if tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]:
                tk_cross_above[i] = True
            elif tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]:
                tk_cross_below[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for Ichimoku calculations
    start_idx = max(52, 1)  # Need 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above cloud + TK cross up
            if (close[i] > upper_cloud[i] and tk_cross_above[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down
            elif (close[i] < lower_cloud[i] and tk_cross_below[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below cloud or TK cross down
            if (close[i] < lower_cloud[i] or tk_cross_below[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or TK cross up
            if (close[i] > upper_cloud[i] or tk_cross_above[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v2"
timeframe = "6h"
leverage = 1.0