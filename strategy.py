#!/usr/bin/env python3
"""
6h_Ichi_Cloud_Kijun_Cross_1dTrend
Hypothesis: 6h Ichimoku Kijun/Tenkan cross with 1d cloud filter (above/below cloud).
The 1d cloud (Senkou Span A/B) acts as a macro trend filter. Only take long when price > cloud,
short when price < cloud. Kijun/Tenkan cross on 6h provides entry timing. Works in bull/bear
by following higher timeframe cloud direction, avoiding counter-trend trades.
"""

name = "6h_Ichi_Cloud_Kijun_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku cloud (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:  # Need 26*2 for Senkou span
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Get 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 periods for Senkou B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud AND Tenkan crosses above Kijun
            if close[i] > cloud_top[i] and tenkan_sen_aligned[i] > kijun_sen_aligned[i] and \
               tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND Tenkan crosses below Kijun
            elif close[i] < cloud_bottom[i] and tenkan_sen_aligned[i] < kijun_sen_aligned[i] and \
                 tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below cloud OR Tenkan crosses below Kijun
            if close[i] < cloud_top[i] or (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                                          tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above cloud OR Tenkan crosses above Kijun
            if close[i] > cloud_bottom[i] or (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                                             tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals