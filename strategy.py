#!/usr/bin/env python3
"""
6h_ichimoku_cloud_1w_trend_v1
Hypothesis: On 6-hour timeframe, use Ichimoku Cloud with weekly trend filter to capture major trends while avoiding whipsaws. The weekly trend (via Ichimoku Cloud on 1w) provides macro context, while the 6h Ichimoku (Tenkan/Kijun cross and cloud position) gives entry timing. This combination works in both bull and bear markets by only taking trades aligned with the weekly trend. Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate 6h Ichimoku
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over last 9 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    tenkan_sen = (high_series.rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  low_series.rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over last 26 periods
    kijun_sen = (high_series.rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 low_series.rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over last 52 periods shifted 26 periods ahead
    senkou_span_b = ((high_series.rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      low_series.rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun_period)
    
    # Get weekly Ichimoku for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < senkou_span_b_period:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    
    tenkan_sen_1w = (high_1w.rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                     low_1w.rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    kijun_sen_1w = (high_1w.rolling(window=kijun_period, min_periods=kijun_period).max() + 
                    low_1w.rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2).shift(kijun_period)
    senkou_span_b_1w = ((high_1w.rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                         low_1w.rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(kijun_period)
    
    # Weekly trend: price above/below cloud
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_1w = np.maximum(senkou_span_a_1w.values, senkou_span_b_1w.values)
    cloud_bottom_1w = np.minimum(senkou_span_a_1w.values, senkou_span_b_1w.values)
    close_1w = df_1w['close'].values
    
    # Weekly trend: 1 = bullish (price above cloud), -1 = bearish (price below cloud), 0 = in cloud
    weekly_trend = np.zeros(len(close_1w))
    weekly_trend[close_1w > cloud_top_1w] = 1
    weekly_trend[close_1w < cloud_bottom_1w] = -1
    
    # Align weekly trend to 6h timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    signals = np.zeros(n)
    
    # Start from enough data for all indicators
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + kijun_period
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or 
            np.isnan(senkou_span_a.iloc[i]) or np.isnan(senkou_span_b.iloc[i]) or
            np.isnan(weekly_trend_aligned[i])):
            continue
        
        # Get current values
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        span_a = senkou_span_a.iloc[i]
        span_b = senkou_span_b.iloc[i]
        close_price = close[i]
        chikou = chikou_span.iloc[i] if not np.isnan(chikou_span.iloc[i]) else close_price
        
        # Cloud boundaries
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Determine if price is above/below cloud
        price_above_cloud = close_price > cloud_top
        price_below_cloud = close_price < cloud_bottom
        
        # Get weekly trend
        trend = weekly_trend_aligned[i]
        
        # Entry conditions
        # Long: weekly bullish + Tenkan crosses above Kijun + price above cloud
        # Short: weekly bearish + Tenkan crosses below Kijun + price below cloud
        
        # Check for crossovers (need previous values)
        if i > 0:
            tenkan_prev = tenkan_sen.iloc[i-1]
            kijun_prev = kijun_sen.iloc[i-1]
            
            tk_cross_up = (tenkan > kijun) and (tenkan_prev <= kijun_prev)
            tk_cross_down = (tenkan < kijun) and (tenkan_prev >= kijun_prev)
        else:
            tk_cross_up = False
            tk_cross_down = False
        
        # Long signal
        if trend == 1 and tk_cross_up and price_above_cloud:
            signals[i] = 0.25
        # Short signal
        elif trend == -1 and tk_cross_down and price_below_cloud:
            signals[i] = -0.25
        # Exit conditions: reverse signal or cloud rejection
        elif (trend == -1 and price_above_cloud) or (trend == 1 and price_below_cloud):
            signals[i] = 0.0
        # Hold previous signal if no change
        else:
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals