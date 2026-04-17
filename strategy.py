#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1D_Cloud_Filter
Strategy: Ichimoku Tenkan/Kijun cross with 1D cloud filter on 6h timeframe.
Long: Tenkan crosses above Kijun + price above 1D cloud (Senkou Span A/B)
Short: Tenkan crosses below Kijun + price below 1D cloud
Exit: Tenkan/Kijun cross reverses or price crosses cloud midpoint
Position size: 0.25
Designed to capture trend momentum with institutional support/resistance levels.
Timeframe: 6h
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
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    highest_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    highest_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_kijun + lowest_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    highest_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_senkou_b + lowest_senkou_b) / 2.0
    
    # Get 1D Ichimoku cloud for filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < senkou_span_b_period:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1D Ichimoku components
    highest_tenkan_1d = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_tenkan_1d = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_1d = (highest_tenkan_1d + lowest_tenkan_1d) / 2.0
    
    highest_kijun_1d = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_kijun_1d = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_1d = (highest_kijun_1d + lowest_kijun_1d) / 2.0
    
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2.0
    
    highest_senkou_b_1d = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_senkou_b_1d = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1d = (highest_senkou_b_1d + lowest_senkou_b_1d) / 2.0
    
    # Align 1D cloud components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Cloud top and bottom (A and B lines)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_middle = (cloud_top + cloud_bottom) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(max(tenkan_period*2, kijun_period*2, senkou_span_b_period*2), n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(cloud_middle[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        tk_cross_up = tenkan[i-1] < kijun[i-1] and tenkan[i] >= kijun[i]  # Tenkan crosses above Kijun
        tk_cross_down = tenkan[i-1] > kijun[i-1] and tenkan[i] <= kijun[i]  # Tenkan crosses below Kijun
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        price_cross_middle_up = close[i-1] < cloud_middle[i-1] and close[i] >= cloud_middle[i]
        price_cross_middle_down = close[i-1] > cloud_middle[i-1] and close[i] <= cloud_middle[i]
        
        # Entry signals
        if position == 0:
            # Long: TK cross up + price above cloud
            if tk_cross_up and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below cloud
            elif tk_cross_down and price_below_cloud:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down or price crosses below cloud middle
            if tk_cross_down or price_cross_middle_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up or price crosses above cloud middle
            if tk_cross_up or price_cross_middle_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_1D_Cloud_Filter"
timeframe = "6h"
leverage = 1.0