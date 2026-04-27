#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h strategy using Ichimoku Tenkan-Kijun cross with cloud filter from 1d timeframe for trend alignment. The Tenkan-Kijun cross provides timely momentum signals, while the 1d cloud acts as a dynamic support/resistance zone and trend filter. This combination reduces false signals in choppy markets and captures strong trends. Designed for BTC/ETH robustness in both bull and bear markets via 1d cloud filter. Targets 50-150 trades over 4 years (12-37/year) with 0.25 position size. Uses discrete levels to minimize fee drag.
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_high_9 + lowest_low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_high_26 + lowest_low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = ((highest_high_52 + lowest_low_52) / 2)
    
    # Get 1d data for cloud filter (Senkou Span A and B from 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Tenkan-sen (9-period)
    highest_high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    lowest_low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (highest_high_9_1d + lowest_low_9_1d) / 2
    
    # 1d Kijun-sen (26-period)
    highest_high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    lowest_low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (highest_high_26_1d + lowest_low_26_1d) / 2
    
    # 1d Senkou Span A
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # 1d Senkou Span B (52-period)
    highest_high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    lowest_low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = ((highest_high_52_1d + lowest_low_52_1d) / 2)
    
    # Align 1d Ichimoku components to 6h timeframe (cloud: Senkou Span A and B)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Calculate cloud boundaries (top and bottom of cloud)
    # In Ichimoku, cloud top is max(Senkou Span A, Senkou Span B), cloud bottom is min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    
    # Determine trend based on 1d cloud: price above cloud = uptrend, below cloud = downtrend
    # We'll use this as a filter: only take longs when price > cloud_top, shorts when price < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 6h Ichimoku (52 for Senkou Span B) + 1d Ichimoku (52) + alignment delay
    # The align_htf_to_ltf function adds delay for completed bars, so we need extra warmup
    start_idx = max(52, 52) + 1  # Ichimoku periods + 1 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        close_val = close[i]
        cloud_top_val = cloud_top[i]
        cloud_bottom_val = cloud_bottom[i]
        
        if position == 0:
            # Look for entry: Tenkan-Kijun cross with 1d cloud filter
            # Bullish cross: Tenkan crosses above Kijun
            # Bearish cross: Tenkan crosses below Kijun
            bullish_cross = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
            bearish_cross = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
            
            # Long condition: bullish cross AND price above 1d cloud (uptrend filter)
            long_condition = bullish_cross and (close_val > cloud_top_val)
            # Short condition: bearish cross AND price below 1d cloud (downtrend filter)
            short_condition = bearish_cross and (close_val < cloud_bottom_val)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun (trend weakening) OR price falls below cloud bottom
            tenkan_kijun_cross_down = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
            price_below_cloud = close_val < cloud_bottom_val
            
            if tenkan_kijun_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun (trend weakening) OR price rises above cloud top
            tenkan_kijun_cross_up = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
            price_above_cloud = close_val > cloud_top_val
            
            if tenkan_kijun_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0