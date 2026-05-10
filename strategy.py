#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_1d
# Hypothesis: Use Ichimoku cloud from daily timeframe to determine trend direction, with price above/below cloud as entry signal. 
# In bull markets, price above cloud indicates uptrend; in bear markets, price below cloud indicates downtrend.
# The cloud acts as dynamic support/resistance, making it effective in both trending and ranging markets.
# Target: 12-37 trades/year on 6h timeframe with strict entry conditions to minimize fee drag.

name = "6h_Ichimoku_Cloud_Trend_1d"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    def highest_n(arr, n):
        res = np.full_like(arr, np.nan)
        for i in range(n-1, len(arr)):
            res[i] = np.max(arr[i-n+1:i+1])
        return res
    
    def lowest_n(arr, n):
        res = np.full_like(arr, np.nan)
        for i in range(n-1, len(arr)):
            res[i] = np.min(arr[i-n+1:i+1])
        return res
    
    tenkan_sen = (highest_n(high_1d, 9) + lowest_n(low_1d, 9)) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (highest_n(high_1d, 26) + lowest_n(low_1d, 26)) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (highest_n(high_1d, 52) + lowest_n(low_1d, 52)) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for daily bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for Ichimoku (max 52 periods)
    
    for i in range(start_idx, n):
        # Skip if any Ichimoku component is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: price above cloud (bullish signal)
            if close[i] > upper_cloud:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud (bearish signal)
            elif close[i] < lower_cloud:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud
            if close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud
            if close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals