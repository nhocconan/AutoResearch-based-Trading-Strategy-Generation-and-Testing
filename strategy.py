#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1wTrendFilter
Hypothesis: Ichimoku TK cross on 6h with 1w trend filter (price above/below weekly cloud) to capture medium-term trends in both bull and bear markets. Uses discrete sizing 0.25 to target 12-37 trades/year. Weekly trend filter reduces false signals during sideways markets.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need enough data for weekly calculations
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Calculate 1w Ichimoku cloud for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Tenkan-sen (9-period)
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2
    
    # Weekly Kijun-sen (26-period)
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2
    
    # Weekly Senkou Span A
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # Weekly Senkou Span B (52-period)
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((high_52_1w + low_52_1w) / 2)
    
    # Align weekly cloud components to 6h timeframe
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Determine weekly trend: price above cloud = uptrend, below cloud = downtrend
    # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
    cloud_top_1w = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    cloud_bottom_1w = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou B, 26 for Kijun
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top_1w[i]) or np.isnan(cloud_bottom_1w[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        close_val = close[i]
        cloud_top = cloud_top_1w[i]
        cloud_bottom = cloud_bottom_1w[i]
        
        if position == 0:
            # Flat - look for TK cross with weekly trend filter
            # Bullish TK cross: Tenkan crosses above Kijun
            tk_bullish = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
            # Bearish TK cross: Tenkan crosses below Kijun
            tk_bearish = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
            
            # Weekly trend filter: only long in uptrend, only short in downtrend
            weekly_uptrend = close_val > cloud_top
            weekly_downtrend = close_val < cloud_bottom
            
            if tk_bullish and weekly_uptrend:
                signals[i] = fixed_size
                position = 1
            elif tk_bearish and weekly_downtrend:
                signals[i] = -fixed_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when TK cross turns bearish or price falls below cloud bottom
            tk_bearish = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
            if tk_bearish or close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit when TK cross turns bullish or price rises above cloud top
            tk_bullish = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
            if tk_bullish or close_val > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1wTrendFilter"
timeframe = "6h"
leverage = 1.0