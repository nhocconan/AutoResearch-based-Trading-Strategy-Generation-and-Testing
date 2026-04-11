#!/usr/bin/env python3
# 6h_1w_1d_ichimoku_cloud_filter_v1
# Strategy: 6h Ichimoku with 1d cloud filter and 1w trend bias
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku provides clear trend direction via cloud and TK cross.
# Use 1d cloud (from daily) to filter trades: only go long when price above 1d cloud,
# short when price below 1d cloud. Use 1w Tenkan/Kijun cross for stronger trend bias.
# This avoids whipsaws in sideways markets and captures strong trends.
# Target: 50-150 trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_ichimoku_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Ichimoku components (for cloud)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for cloud)
    
    # Align 1d Ichimoku components to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d.values)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d.values)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d.values)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d.values)
    
    # 1w Ichimoku for trend bias (Tenkan/Kijun cross)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    
    # Align 1w Ichimoku components to 6h
    tenkan_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_1w.values)
    kijun_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_1w.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Need enough data for Ichimoku calculations
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or
            np.isnan(tenkan_sen_1w_aligned[i]) or np.isnan(kijun_sen_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        cloud_bottom = min(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # Price relative to 1d cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # 1w trend bias: Tenkan/Kijun cross
        # Bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
        w_bullish = tenkan_sen_1w_aligned[i] > kijun_sen_1w_aligned[i]
        w_bearish = tenkan_sen_1w_aligned[i] < kijun_sen_1w_aligned[i]
        
        # Entry logic: price outside cloud + 1w trend alignment
        if price_above_cloud and w_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        elif price_below_cloud and w_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price re-enters cloud or trend changes
        elif position == 1 and (not price_above_cloud or not w_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not price_below_cloud or not w_bearish):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals