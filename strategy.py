#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: Use Ichimoku cloud from 1d timeframe to determine trend direction, with 6h Tenkan/Kijun cross for entry timing. 
Long when price above 1d cloud + 6h Tenkan crosses above Kijun. Short when price below 1d cloud + 6h Tenkan crosses below Kijun.
Exit when price crosses opposite cloud boundary or Tenkan/Kijun reverse. 
Ichimoku works in both bull (trend following) and bear (counter-trend reversals) markets by capturing momentum shifts.
Designed for 6h timeframe to limit trades (target: 50-150 total over 4 years) and avoid fee drag.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
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
    
    # Get daily data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        top_cloud = max(span_a_aligned[i], span_b_aligned[i])
        bottom_cloud = min(span_a_aligned[i], span_b_aligned[i])
        
        if position == 0:
            # LONG: Price above cloud + Tenkan crosses above Kijun
            if close[i] > top_cloud and tenkan_aligned[i-1] <= kijun_aligned[i-1] and tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud + Tenkan crosses below Kijun
            elif close[i] < bottom_cloud and tenkan_aligned[i-1] >= kijun_aligned[i-1] and tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below cloud OR Tenkan crosses below Kijun
            if close[i] < bottom_cloud or (tenkan_aligned[i-1] >= kijun_aligned[i-1] and tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above cloud OR Tenkan crosses above Kijun
            if close[i] > top_cloud or (tenkan_aligned[i-1] <= kijun_aligned[i-1] and tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals