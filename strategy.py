#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud color filter from 1d timeframe. 
Enter long when TK cross bullish + price above cloud (bullish regime), short when TK cross bearish + price below cloud (bearish regime).
Uses discrete sizing 0.25 to limit trades (~15-30/year). Works in bull/bear via 1d cloud filter as regime.
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
    
    # Load 1d data ONCE before loop for HTF regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 for Senkou Span B
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        size = fixed_size
        
        # Determine cloud color and boundaries
        if span_a > span_b:
            # Bullish cloud: green
            cloud_top = span_a
            cloud_bottom = span_b
            is_bullish_cloud = True
        else:
            # Bearish cloud: red
            cloud_top = span_b
            cloud_bottom = span_a
            is_bullish_cloud = False
        
        # TK cross signals
        tk_bullish_cross = tenkan_val > kijun_val and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_bearish_cross = tenkan_val < kijun_val and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 0:
            # Flat - look for entry
            # Long: bullish TK cross + price above cloud + bullish cloud
            # Short: bearish TK cross + price below cloud + bearish cloud
            long_entry = tk_bullish_cross and close_val > cloud_top and is_bullish_cloud
            short_entry = tk_bearish_cross and close_val < cloud_bottom and not is_bullish_cloud
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on bearish TK cross or price drops below cloud
            if tk_bearish_cross or close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on bullish TK cross or price rises above cloud
            if tk_bullish_cross or close_val > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrendFilter"
timeframe = "6h"
leverage = 1.0