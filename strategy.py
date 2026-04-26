#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1d
Hypothesis: Ichimoku Tenkan-Kijun cross with 1d cloud filter on 6h timeframe.
Long when Tenkan crosses above Kijun and price is above 1d cloud (bullish regime).
Short when Tenkan crosses below Kijun and price is below 1d cloud (bearish regime).
Uses discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear markets
by following the 1d Ichimoku cloud as trend filter. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 52 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Calculate Ichimoku components on 1d for cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    # But we need the cloud values for current time, so we calculate:
    # Senkou Span A = (1d Tenkan + 1d Kijun) / 2
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # The cloud (Kumo) is between Senkou Span A and Senkou Span B
    # For current time, we use the values as calculated (no shift needed for alignment)
    # Upper cloud = max(Senkou Span A, Senkou Span B)
    # Lower cloud = min(Senkou Span A, Senkou Span B)
    upper_cloud_1d = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    lower_cloud_1d = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    # Align 1d cloud to 6h timeframe
    upper_cloud_aligned = align_htf_to_ltf(prices, df_1d, upper_cloud_1d)
    lower_cloud_aligned = align_htf_to_ltf(prices, df_1d, lower_cloud_1d)
    
    # TK cross signals (we need to detect crossovers)
    # Bullish cross: Tenkan crosses above Kijun
    # Bearish cross: Tenkan crosses below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    # Handle first element
    tk_cross_up[0] = False
    tk_cross_down[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou Span B calculation)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(upper_cloud_aligned[i]) or np.isnan(lower_cloud_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above 1d cloud (bullish regime)
            if tk_cross_up[i] and close[i] > upper_cloud_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below 1d cloud (bearish regime)
            elif tk_cross_down[i] and close[i] < lower_cloud_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Tenkan crosses below Kijun (trend change) OR price falls below cloud
            if tk_cross_down[i] or close[i] < lower_cloud_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Tenkan crosses above Kijun (trend change) OR price rises above cloud
            if tk_cross_up[i] or close[i] > upper_cloud_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1d"
timeframe = "6h"
leverage = 1.0