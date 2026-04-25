#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1wTrend
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) with cloud filter from 1d and trend filter from 1w.
Long when TK crosses above Kijun, price above cloud, and 1w bullish (price > 1w Kumo Senkou Span B).
Short when TK crosses below Kijun, price below cloud, and 1w bearish (price < 1w Kumo Senkou Span B).
Uses discrete sizing (0.25) to limit trades (~12-37/year on 6h) and minimize fee drag.
Designed to work in both bull and bear markets via multi-timeframe trend alignment.
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
    
    # Ichimoku parameters (6h timeframe)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    highest_high_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_high_tenkan + lowest_low_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    highest_high_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_high_kijun + lowest_low_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan + kijun) / 2)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26 periods
    highest_high_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_high_senkou_b + lowest_low_senkou_b) / 2
    
    # 1d data for cloud filter (current cloud)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku cloud (using same parameters)
    highest_high_1d_tenkan = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_low_1d_tenkan = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_1d = (highest_high_1d_tenkan + lowest_low_1d_tenkan) / 2
    
    highest_high_1d_kijun = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_low_1d_kijun = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_1d = (highest_high_1d_kijun + lowest_low_1d_kijun) / 2
    
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    highest_high_1d_senkou_b = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_1d_senkou_b = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1d = (highest_high_1d_senkou_b + lowest_low_1d_senkou_b) / 2
    
    # 1w data for trend filter (Kumo Senkou Span B)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Ichimoku Senkou Span B for trend filter
    highest_high_1w_senkou_b = pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_low_1w_senkou_b = pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1w = (highest_high_1w_senkou_b + lowest_low_1w_senkou_b) / 2
    
    # Align all HTF indicators to 6h timeframe (completed bar only)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need all Ichimoku calculations to be ready
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 26  # +26 for Senkou Span shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or
            np.isnan(senkou_span_b_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # 1d cloud filter
        upper_cloud_1d = max(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        lower_cloud_1d = min(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # 1w trend filter: bullish if price > 1w Senkou Span B, bearish if price < 1w Senkou Span B
        trend_bullish = curr_close > senkou_span_b_1w_aligned[i]
        trend_bearish = curr_close < senkou_span_b_1w_aligned[i]
        
        if position == 0:
            # Long: TK cross above, price above cloud (6h and 1d), 1w bullish trend
            tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud_6h = curr_close > upper_cloud
            price_above_cloud_1d = curr_close > upper_cloud_1d
            
            if tk_cross_above and price_above_cloud_6h and price_above_cloud_1d and trend_bullish:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: TK cross below OR price below cloud (6h) OR 1w trend turns bearish
            tk_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
            price_below_cloud = curr_close < lower_cloud
            
            if tk_cross_below or price_below_cloud or not trend_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: TK cross above OR price above cloud (6h) OR 1w trend turns bullish
            tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
            price_above_cloud = curr_close > upper_cloud
            
            if tk_cross_above or price_above_cloud or trend_bullish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend"
timeframe = "6h"
leverage = 1.0