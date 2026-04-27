#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: Use Ichimoku cloud (conversion/base lines + leading span A/B) from 1d timeframe to define trend and support/resistance. Enter long when price breaks above cloud with bullish TK cross, short when price breaks below cloud with bearish TK cross. Exit when price re-enters cloud. Uses 1d Ichimoku for higher timeframe structure to avoid whipsaws in choppy 6h markets. Target 15-30 trades/year to minimize fee drift while capturing major trend moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: conversion line, base line, leading span A, leading span B"""
    # Conversion line (Tenkan-sen): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    conversion_line = (period9_high + period9_low) / 2
    
    # Base line (Kijun-sen): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    base_line = (period26_high + period26_low) / 2
    
    # Leading span A (Senkou span A): (conversion line + base line)/2 shifted 26 periods ahead
    leading_span_a = ((conversion_line + base_line) / 2)
    
    # Leading span B (Senkou span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    leading_span_b = (period52_high + period52_low) / 2
    
    return conversion_line, base_line, leading_span_a, leading_span_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d Ichimoku data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    conv_line, base_line, lead_span_a, lead_span_b = calculate_ichimoku(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe (wait for 1d close)
    conv_line_aligned = align_htf_to_ltf(prices, df_1d, conv_line)
    base_line_aligned = align_htf_to_ltf(prices, df_1d, base_line)
    lead_span_a_aligned = align_htf_to_ltf(prices, df_1d, lead_span_a)
    lead_span_b_aligned = align_htf_to_ltf(prices, df_1d, lead_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(conv_line_aligned[i]) or np.isnan(base_line_aligned[i]) or 
            np.isnan(lead_span_a_aligned[i]) or np.isnan(lead_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (using already shifted values from Ichimoku calculation)
        cloud_top = max(lead_span_a_aligned[i], lead_span_b_aligned[i])
        cloud_bottom = min(lead_span_a_aligned[i], lead_span_b_aligned[i])
        
        # TK cross signals
        tk_bullish = conv_line_aligned[i] > base_line_aligned[i]
        tk_bearish = conv_line_aligned[i] < base_line_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud with bullish TK cross
            if close[i] > cloud_top and tk_bullish:
                signals[i] = size
                position = 1
            # Short: price breaks below cloud with bearish TK cross
            elif close[i] < cloud_bottom and tk_bearish:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud (breaks below cloud top)
            if close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters cloud (breaks above cloud bottom)
            if close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0