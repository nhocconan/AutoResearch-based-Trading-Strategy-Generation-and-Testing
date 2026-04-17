#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1-week Ichimoku Cloud (Tenkan/Kijun/Senkou) filter.
Long when price > Senkou Span B and Tenkan > Kijun; short when opposite.
Uses 1-week data for trend structure to avoid whipsaws in 6h timeframe.
Targets 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
Works in bull/bear by following higher timeframe trend via Ichimoku cloud.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1w data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    # Senkou Span A: (Tenkan + Kijun)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B: (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Displace Senkou spans forward by 26 periods
    senkou_span_a = senkou_span_a.shift(displacement)
    senkou_span_b = senkou_span_b.shift(displacement)
    
    # Align to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above Senkou Span B and Tenkan > Kijun
            if close[i] > senkou_span_b_aligned[i] and tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below Senkou Span B and Tenkan < Kijun
            elif close[i] < senkou_span_b_aligned[i] and tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Senkou Span B or Tenkan < Kijun
            if close[i] < senkou_span_b_aligned[i] or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Senkou Span B or Tenkan > Kijun
            if close[i] > senkou_span_b_aligned[i] or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wIchimoku_CloudFilter"
timeframe = "6h"
leverage = 1.0