#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: Ichimoku Tenkan-Kijun cross with cloud filter from 1d timeframe on 6h chart.
Works in bull markets (price above cloud, bullish TK cross) and bear markets (price below cloud, bearish TK cross).
Uses 1d Ichimoku cloud as regime filter to avoid counter-trend trades. Targets 12-30 trades/year.
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
    
    # Get 1d data for Ichimoku cloud (senkou span A/B) and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                 pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2).shift(displacement)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                         pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(displacement)
    
    # Align Ichimoku components to 6h timeframe (completed 1d bar only)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # Determine cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # 1d trend filter: price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku components
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + displacement
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above cloud + Tenkan crosses above Kijun (bullish TK cross)
            bullish_tk_cross = (tenkan_1d_aligned[i] > kijun_1d_aligned[i]) and \
                               (tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1])
            long_setup = price_above_cloud[i] and bullish_tk_cross
            
            # Short: price below cloud + Tenkan crosses below Kijun (bearish TK cross)
            bearish_tk_cross = (tenkan_1d_aligned[i] < kijun_1d_aligned[i]) and \
                               (tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1])
            short_setup = price_below_cloud[i] and bearish_tk_cross
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price crosses below cloud OR Tenkan crosses below Kijun
            if (close[i] < cloud_top[i]) or \
               (tenkan_1d_aligned[i] < kijun_1d_aligned[i] and tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price crosses above cloud OR Tenkan crosses above Kijun
            if (close[i] > cloud_bottom[i]) or \
               (tenkan_1d_aligned[i] > kijun_1d_aligned[i] and tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0