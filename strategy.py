#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend
Hypothesis: Ichimoku TK cross with cloud filter from 1d trend provides high-probability entries on 6h timeframe.
Works in both bull and bear markets by using 1d cloud as trend filter and TK cross for timing.
Target: 60-120 trades over 4 years (15-30/year) with disciplined risk control.
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
    
    # Get 1d data for Ichimoku calculations (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe (with proper delay for forward-looking spans)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_cross_bull = tenkan > kijun
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_cross_bear = tenkan < kijun
        
        if position == 0:
            # Long: bullish TK cross above cloud with bullish cloud (span A > span B)
            if tk_cross_bull and price > cloud_top and span_a > span_b:
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross below cloud with bearish cloud (span A < span B)
            elif tk_cross_bear and price < cloud_bottom and span_a < span_b:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: bearish TK cross or price drops below cloud bottom
            if tk_cross_bear or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: bullish TK cross or price rises above cloud top
            if tk_cross_bull or price > cloud_top:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0