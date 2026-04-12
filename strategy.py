#!/usr/bin/env python3
"""
6h_12h_ichimoku_cloud_v1
Hypothesis: 6-hour strategy using 12-hour Ichimoku Cloud for trend direction and entry signals.
The strategy takes long positions when price is above the cloud and Tenkan-sen crosses above Kijun-sen,
and short positions when price is below the cloud and Tenkan-sen crosses below Kijun-sen.
Ichimoku is a comprehensive trend system that adapts to both trending and ranging markets,
providing clear entry/exit signals with built-in trend filtering.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in bull/bear by requiring price to be outside the cloud (strong trend) and using TK cross for timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for Ichimoku
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (standard parameters: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Check for Tenkan/Kijun cross
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Long entry: price above cloud AND TK cross up
        if (close[i] > upper_cloud and tk_cross_up and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price below cloud AND TK cross down
        elif (close[i] < lower_cloud and tk_cross_down and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price enters the cloud or reverse TK cross
        elif position == 1 and (close[i] < upper_cloud or tk_cross_down):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > lower_cloud or tk_cross_up):
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

name = "6h_12h_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0