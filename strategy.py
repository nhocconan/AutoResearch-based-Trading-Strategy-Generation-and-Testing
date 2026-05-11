#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Follow_v1
Hypothesis: Use Ichimoku cloud (conversion/base lines + leading span A/B) from 1d for trend direction,
combined with 6h price above/below cloud and TK cross for entry. The Ichimoku cloud acts as
dynamic support/resistance and trend filter. In bull markets, price stays above cloud; in bear
markets, price stays below cloud. TK cross provides timely entries within the trend.
Target: 50-150 total trades over 4 years on 6h timeframe.
"""

name = "6h_Ichimoku_Cloud_Trend_Follow_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D Data for Ichimoku Cloud ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku
        return np.zeros(n)
    
    # Ichimoku parameters: conversion line (9), base line (26), leading span B (52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Conversion line (Tenkan-sen): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    conversion_line = (period9_high + period9_low) / 2
    
    # Base line (Kijun-sen): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    base_line = (period26_high + period26_low) / 2
    
    # Leading span A (Senkou span A): (Conversion line + Base line) / 2
    leading_span_a = (conversion_line + base_line) / 2
    
    # Leading span B (Senkou span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    leading_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    conversion_aligned = align_htf_to_ltf(prices, df_1d, conversion_line)
    base_aligned = align_htf_to_ltf(prices, df_1d, base_line)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, leading_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, leading_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need at least 52 days of data)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(conversion_aligned[i]) or 
            np.isnan(base_aligned[i]) or 
            np.isnan(span_a_aligned[i]) or 
            np.isnan(span_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(span_a_aligned[i], span_b_aligned[i])
        cloud_bottom = min(span_a_aligned[i], span_b_aligned[i])
        
        # Determine trend: price above cloud = uptrend, below cloud = downtrend
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # TK cross: conversion line crossing base line
        tk_cross_bull = conversion_aligned[i] > base_aligned[i]
        tk_cross_bear = conversion_aligned[i] < base_aligned[i]
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross
            if price_above_cloud and tk_cross_bull:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND bearish TK cross
            elif price_below_cloud and tk_cross_bear:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below cloud OR bearish TK cross
            if not price_above_cloud or tk_cross_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above cloud OR bullish TK cross
            if not price_below_cloud or tk_cross_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals