#!/usr/bin/env python3
"""
6h_1w_1d_ichimoku_cloud_trend
Hypothesis: 6-hour strategy using Ichimoku Cloud on weekly timeframe for trend direction,
with Tenkan-Kijun cross on daily timeframe for entry timing and price above/below cloud for filtering.
Ichimoku provides strong trend identification and support/resistance levels.
Works in both bull and bear markets by only taking trades aligned with weekly trend.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Ichimoku trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku on weekly data
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Get daily data for TK cross
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on daily data for TK cross
    tenkan_1d, kijun_1d, _, _ = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Align daily TK cross to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below cloud
        cloud_top = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Daily TK cross
        tk_cross_up = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_down = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Entry conditions
        if price_above_cloud and tk_cross_up and position != 1:
            # Long: price above weekly cloud + TK cross up
            position = 1
            signals[i] = 0.25
        elif price_below_cloud and tk_cross_down and position != -1:
            # Short: price below weekly cloud + TK cross down
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite TK cross or price crosses cloud
        elif position == 1 and (tk_cross_down or price_below_cloud):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tk_cross_up or price_above_cloud):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_ichimoku_cloud_trend"
timeframe = "6h"
leverage = 1.0