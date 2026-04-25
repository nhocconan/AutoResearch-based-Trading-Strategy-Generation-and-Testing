#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wTrend_Filter
Hypothesis: On 6h timeframe, Ichimoku Tenkan-Kijun (TK) cross with 1w trend filter (price above/below weekly Kumo) captures medium-term momentum with low trade frequency. The 1w Kumo acts as a strong dynamic support/resistance zone, filtering out false TK crosses during sideways markets. Works in both bull and bear markets by aligning with higher timeframe trend. Target: 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 1w data for HTF trend filter (weekly Kumo)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations (52 periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        close_val = close[i]
        
        # Determine Kumo (cloud) boundaries
        upper_kumo = max(senkou_a_val, senkou_b_val)
        lower_kumo = min(senkou_a_val, senkou_b_val)
        
        # TK cross conditions
        tk_cross_up = tenkan_val > kijun_val
        tk_cross_down = tenkan_val < kijun_val
        
        # Price relative to Kumo
        price_above_kumo = close_val > upper_kumo
        price_below_kumo = close_val < lower_kumo
        
        if position == 0:
            # Look for entry signals: TK cross with price aligned with Kumo
            # Long: TK cross up AND price above Kumo (bullish alignment)
            if tk_cross_up and price_above_kumo:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND price below Kumo (bearish alignment)
            elif tk_cross_down and price_below_kumo:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. TK cross down (exit long)
            if tk_cross_down:
                signals[i] = 0.0
                position = 0
            # 2. Price falls below Kumo (exit long)
            elif close_val < upper_kumo:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. TK cross up (exit short)
            if tk_cross_up:
                signals[i] = 0.0
                position = 0
            # 2. Price rises above Kumo (exit short)
            elif close_val > lower_kumo:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0