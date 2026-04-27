#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend
Hypothesis: 6h strategy using Ichimoku Tenkan-Kijun cross with 1d cloud filter. The Ichimoku system provides multiple confirmation lines (Tenkan, Kijun, Senkou Span A/B) that work well in both trending and ranging markets. Using 1d cloud (Senkou Span) as a trend filter ensures we only take trades aligned with the higher timeframe momentum. TK cross provides timely entries while cloud acts as dynamic support/resistance. Designed for BTC/ETH robustness with discrete position sizing to minimize fee drag. Targets 50-150 trades over 4 years.
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
    
    # Ichimoku parameters (standard: 9, 26, 52)
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
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 periods ahead
    highest_high_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_high_52 + lowest_low_52) / 2
    
    # Get 1d data for cloud filter (Senkou Span A/B from 1d)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku cloud
    highest_high_1d_9 = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    lowest_low_1d_9 = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (highest_high_1d_9 + lowest_low_1d_9) / 2
    
    highest_high_1d_26 = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    lowest_low_1d_26 = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_1d = (highest_high_1d_26 + lowest_low_1d_26) / 2
    
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    highest_high_1d_52 = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    lowest_low_1d_52 = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (highest_high_1d_52 + lowest_low_1d_52) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # The cloud is between Senkou Span A and B
    # In an uptrend: Senkou Span A > Senkou Span B
    # In a downtrend: Senkou Span A < Senkou Span B
    cloud_top_1d = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 52 periods for Senkou Span B
    start_idx = senkou_span_b_period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i])):
            signals[i] = 0.0
            continue
        
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        close_val = close[i]
        cloud_top = cloud_top_1d[i]
        cloud_bottom = cloud_bottom_1d[i]
        
        if position == 0:
            # Look for entry: TK cross with price above/below cloud
            # Bullish: Tenkan crosses above Kijun AND price above cloud
            bullish_cross = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            price_above_cloud = close_val > cloud_top
            
            # Bearish: Tenkan crosses below Kijun AND price below cloud
            bearish_cross = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            price_below_cloud = close_val < cloud_bottom
            
            if bullish_cross and price_above_cloud:
                signals[i] = size
                position = 1
            elif bearish_cross and price_below_cloud:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price falls below cloud
            bearish_cross = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
            price_below_cloud = close_val < cloud_bottom
            
            if bearish_cross or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price rises above cloud
            bullish_cross = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
            price_above_cloud = close_val > cloud_top
            
            if bullish_cross or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend"
timeframe = "6h"
leverage = 1.0