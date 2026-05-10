#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_Cloud_Filter_1d
# Hypothesis: Ichimoku Tenkan-Kijun cross with daily cloud filter for high-probability trend entries.
# Tenkan (9) and Kijun (26) crosses signal momentum shifts; price above/below daily Kumo (cloud) filters direction.
# Works in bull/bear by aligning with daily trend via cloud position. Targets 50-150 trades over 4 years.
# Position size 0.25 for balanced risk management.

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1d"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Senkou B
        return np.zeros(n)
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((highest_senkou_b + lowest_senkou_b) / 2)
    
    # Calculate daily Ichimoku components for cloud filter
    # Daily Tenkan-sen (9-period)
    d_highest_tenkan = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    d_lowest_tenkan = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    d_tenkan = (d_highest_tenkan + d_lowest_tenkan) / 2
    
    # Daily Kijun-sen (26-period)
    d_highest_kijun = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    d_lowest_kijun = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    d_kijun = (d_highest_kijun + d_lowest_kijun) / 2
    
    # Daily Senkou Span A
    d_senkou_a = ((d_tenkan + d_kijun) / 2)
    
    # Daily Senkou Span B (52-period)
    d_highest_senkou_b = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    d_lowest_senkou_b = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    d_senkou_b = ((d_highest_senkou_b + d_lowest_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), tenkan)
    kijun_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), senkou_b)
    
    # Align daily Ichimoku components to 6t timeframe
    d_tenkan_aligned = align_htf_to_ltf(prices, df_1d, d_tenkan)
    d_kijun_aligned = align_htf_to_ltf(prices, df_1d, d_kijun)
    d_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_a)
    d_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, d_senkou_b)
    
    # Determine cloud boundaries (Senkou Span A and B)
    # Upper cloud = max(Senkou A, Senkou B), Lower cloud = min(Senkou A, Senkou B)
    d_cloud_top = np.maximum(d_senkou_a_aligned, d_senkou_b_aligned)
    d_cloud_bottom = np.minimum(d_senkou_a_aligned, d_senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 52)  # Warmup for Kijun and Senkou B
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(d_tenkan_aligned[i]) or np.isnan(d_kijun_aligned[i]) or
            np.isnan(d_cloud_top[i]) or np.isnan(d_cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TK Cross signals
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price relative to daily cloud
        price_above_cloud = close[i] > d_cloud_top[i]
        price_below_cloud = close[i] < d_cloud_bottom[i]
        
        if position == 0:
            # Long entry: TK cross up AND price above daily cloud (bullish alignment)
            if tk_cross_up and price_above_cloud:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross down AND price below daily cloud (bearish alignment)
            elif tk_cross_down and price_below_cloud:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down OR price falls below daily cloud
            if tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price rises above daily cloud
            if tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals