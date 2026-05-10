#!/usr/bin/env python3
# 6h_Ichimoku_Kijun_Tenkan_Cross_1dCloud_Trend
# Hypothesis: Ichimoku Kijun/Tenkan cross with 1d cloud filter provides high-probability entries aligned with higher timeframe trend.
# In trending markets, Tenkan crossing above/below Kijun signals momentum shifts; cloud acts as dynamic support/resistance.
# Works in both bull and bear markets by only taking trades in direction of 1d cloud (trend filter).
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_Ichimoku_Kijun_Tenkan_Cross_1dCloud_Trend"
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
    
    # 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but needed for cloud calculation
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Trend filter: price above/below cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Tenkan/Kijun cross signals
    tenkan_above_kijun = tenkan_sen_aligned > kijun_sen_aligned
    tenkan_below_kijun = tenkan_sen_aligned < kijun_sen_aligned
    
    # Previous cross for crossover detection
    tenkan_above_kijun_prev = np.roll(tenkan_above_kijun, 1)
    tenkan_below_kijun_prev = np.roll(tenkan_below_kijun, 1)
    tenkan_above_kijun_prev[0] = False
    tenkan_below_kijun_prev[0] = False
    
    # Bullish cross: Tenkan crosses above Kijun
    bullish_cross = tenkan_above_kijun & ~tenkan_above_kijun_prev
    # Bearish cross: Tenkan crosses below Kijun
    bearish_cross = tenkan_below_kijun & ~tenkan_below_kijun_prev
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish cross + price above cloud (uptrend)
            if bullish_cross[i] and price_above_cloud[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish cross + price below cloud (downtrend)
            elif bearish_cross[i] and price_below_cloud[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Tenkan crosses below Kijun OR price drops below cloud
            if (tenkan_below_kijun[i] or price_below_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan_above_kijun[i] or price_above_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals