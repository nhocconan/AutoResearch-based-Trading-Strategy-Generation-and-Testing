#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Ichimoku TK Cross + Cloud Filter from 1d
    # Long: TK Cross bullish + price above 1d Kumo cloud
    # Short: TK Cross bearish + price below 1d Kumo cloud
    # Exit: TK Cross reversal OR price crosses Kumo cloud
    # Target: 50-150 total trades over 4 years (12-37/year) for low fee drag
    # Works in bull via long bias from cloud, in bear via short bias from cloud
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components (1d)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    # Chikou Span (Lagging Span): not used in this strategy
    
    # Calculate Tenkan-sen
    tenkan_sen = np.full(len(close_1d), np.nan)
    for i in range(period_tenkan - 1, len(close_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Calculate Kijun-sen
    kijun_sen = np.full(len(close_1d), np.nan)
    for i in range(period_kijun - 1, len(close_1d)):
        kijun_sen[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Calculate Senkou Span A
    senkou_span_a = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B
    senkou_span_b = np.full(len(close_1d), np.nan)
    for i in range(period_senkou_b - 1, len(close_1d)):
        senkou_span_b[i] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # TK Cross conditions
        tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        # Cloud conditions (price above/below cloud)
        # Cloud top is max(Senkou Span A, Senkou Span B)
        # Cloud bottom is min(Senkou Span A, Senkou Span B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Entry logic: TK Cross + price relative to cloud
        long_entry = tk_bullish and price_above_cloud
        short_entry = tk_bearish and price_below_cloud
        
        # Exit logic: TK Cross reversal OR price crosses cloud
        long_exit = tk_bearish or close[i] < cloud_top
        short_exit = tk_bullish or close[i] > cloud_bottom
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ichimoku_tk_cross_cloud_filter_v1"
timeframe = "6h"
leverage = 1.0