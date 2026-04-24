#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with TK Cross and cloud filter from 1d timeframe.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Ichimoku cloud (Senkou Span A/B) and trend filter.
- Entry: Long when TK cross (Tenkan/Kijun) occurs above cloud AND price > cloud;
         Short when TK cross occurs below cloud AND price < cloud.
- Exit: Close-based reversal (opposite TK cross) or when price crosses cloud in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Ichimoku provides dynamic support/resistance and trend direction; works in bull via cloud breakouts and bear via cloud rejection.
- Weekly pivot from 1w adds bias: only take longs above weekly pivot, shorts below.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2.0
    
    # Get 1d data for cloud filter (Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < period_senkou_b:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku components on 1d
    tenkan_1d = (pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values +
                 pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2.0
    kijun_1d = (pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values +
                pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2.0
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2.0
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values +
                        pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2.0
    
    # Get 1w data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    # Use previous week's values (shift by 1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    weekly_pivot[0] = np.nan  # First value has no previous week
    
    # Align all indicators to primary 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period_senkou_b, period_kijun)  # 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price = close[i]
        
        # TK cross signals
        tk_cross_bull = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_bear = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Cloud filter: price above/below cloud
        price_above_cloud = price > upper_cloud
        price_below_cloud = price < lower_cloud
        
        # Weekly pivot bias
        price_above_weekly_pivot = price > weekly_pivot_aligned[i]
        price_below_weekly_pivot = price < weekly_pivot_aligned[i]
        
        if position == 0:
            # Check for entry signals with cloud filter and weekly pivot bias
            if tk_cross_bull and price_above_cloud and price_above_weekly_pivot:
                signals[i] = 0.25
                position = 1
            elif tk_cross_bear and price_below_cloud and price_below_weekly_pivot:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when TK cross bearish OR price crosses below cloud
            if tk_cross_bear or price < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when TK cross bullish OR price crosses above cloud
            if tk_cross_bull or price > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dCloudFilter_1wPivotBias_v1"
timeframe = "6h"
leverage = 1.0