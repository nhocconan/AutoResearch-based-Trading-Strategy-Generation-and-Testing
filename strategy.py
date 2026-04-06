#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Volume Confirmation
Hypothesis: Ichimoku provides multi-dimensional trend analysis (support/resistance, momentum, direction). 
Cloud acts as dynamic support/resistance. TK cross gives entry signal. Volume confirms institutional participation.
Works in bull (price above cloud, bullish TK cross) and bear (price below cloud, bearish TK cross) markets.
Designed for 50-150 trades over 4 years (12-37/year) on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Ichimoku (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku calculation on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_9 = np.full_like(high_1d, np.nan)
    min_low_9 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            max_high_9[i] = np.max(high_1d[i - period_tenkan + 1:i + 1])
            min_low_9[i] = np.min(low_1d[i - period_tenkan + 1:i + 1])
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_26 = np.full_like(high_1d, np.nan)
    min_low_26 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1:
            max_high_26[i] = np.max(high_1d[i - period_kijun + 1:i + 1])
            min_low_26[i] = np.min(low_1d[i - period_kijun + 1:i + 1])
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = np.full_like(high_1d, np.nan)
    min_low_52 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_senkou_b - 1:
            max_high_52[i] = np.max(high_1d[i - period_senkou_b + 1:i + 1])
            min_low_52[i] = np.min(low_1d[i - period_senkou_b + 1:i + 1])
    senkou_span_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period (need 52 periods for Senkou Span B)
    start = 52 + 26  # 52 for calculation + 26 for shift
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK Cross
        tk_cross = tenkan_sen_aligned[i] - kijun_sen_aligned[i]
        tk_cross_prev = tenkan_sen_aligned[i-1] - kijun_sen_aligned[i-1] if i > 0 else 0
        
        bullish_tk_cross = tk_cross > 0 and tk_cross_prev <= 0
        bearish_tk_cross = tk_cross < 0 and tk_cross_prev >= 0
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price goes below cloud OR bearish TK cross
            if close[i] < lower_cloud or bearish_tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price goes above cloud OR bullish TK cross
            if close[i] > upper_cloud or bullish_tk_cross:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price relative to cloud + TK cross + volume
            price_above_cloud = close[i] > upper_cloud
            price_below_cloud = close[i] < lower_cloud
            
            if i >= 20 and price_above_cloud and bullish_tk_cross and volume_filter:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and price_below_cloud and bearish_tk_cross and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals