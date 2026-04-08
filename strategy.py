#!/usr/bin/env python3
# 6h_weekly_ichimoku_trend_follow_v1
# Hypothesis: 6h trend following using weekly Ichimoku cloud as primary filter with 1d TK cross for entry timing works in both bull and bear markets.
# Long: price above weekly cloud + 1d Tenkan crosses above Kijun
# Short: price below weekly cloud + 1d Tenkan crosses below Kijun
# Exit: opposite TK cross or price re-enters the cloud
# Uses 6h primary timeframe with 1w HTF for cloud and 1d HTF for TK cross.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for Ichimoku cloud (Senkou Span A/B)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for proper calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2.0)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    high_52_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((high_52_1w + low_52_1w) / 2.0)
    
    # Align weekly Ichimoku components to 6h timeframe (with proper look-ahead prevention)
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Get 1d data for TK cross (Tenkan/Kijun cross)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data for TK cross
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2.0
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2.0
    
    # Align daily TK cross to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        lower_cloud = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        if position == 1:  # Long position
            # Exit: Tenkan crosses below Kijun OR price re-enters cloud (below upper cloud)
            if tenkan_1d_aligned[i] < kijun_1d_aligned[i] or price < upper_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan crosses above Kijun OR price re-enters cloud (above lower cloud)
            if tenkan_1d_aligned[i] > kijun_1d_aligned[i] or price > lower_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price above cloud AND Tenkan crosses above Kijun
            if price > upper_cloud and tenkan_1d_aligned[i] > kijun_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below cloud AND Tenkan crosses below Kijun
            elif price < lower_cloud and tenkan_1d_aligned[i] < kijun_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals