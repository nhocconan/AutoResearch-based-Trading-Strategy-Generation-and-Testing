#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter
Hypothesis: Ichimoku Tenkan-Kijun cross combined with cloud filter (price above/below cloud) on 6h timeframe, filtered by 1d Ichimoku trend direction, captures high-probability trend continuations while avoiding counter-trend whipsaws. Works in bull/bear by following 1d trend direction.
"""

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou_a, senkou_b"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on 6h data
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h = calculate_ichimoku(high, low, close)
    
    # Calculate Ichimoku on 1d data for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom (Senkou A and B)
    cloud_top_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # 1d trend: price above/below cloud
    bullish_trend_1d = close_1d > np.maximum(senkou_a_1d, senkou_b_1d)
    bearish_trend_1d = close_1d < np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d trend to 6h
    bullish_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend_1d.astype(float))
    bearish_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top_6h[i]) or np.isnan(cloud_bottom_6h[i]) or
            np.isnan(bullish_trend_1d_aligned[i]) or np.isnan(bearish_trend_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # TK cross signals
        tk_cross_bullish = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_bearish = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        # Price position relative to cloud
        price_above_cloud = close[i] > cloud_top_6h[i]
        price_below_cloud = close[i] < cloud_bottom_6h[i]
        
        if position == 0:
            # LONG: Bullish TK cross + price above cloud + 1d bullish trend
            if tk_cross_bullish and price_above_cloud and bullish_trend_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish TK cross + price below cloud + 1d bearish trend
            elif tk_cross_bearish and price_below_cloud and bearish_trend_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish TK cross or price below cloud
            if tk_cross_bearish or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish TK cross or price above cloud
            if tk_cross_bullish or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals