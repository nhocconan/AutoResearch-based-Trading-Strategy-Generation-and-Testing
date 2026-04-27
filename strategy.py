#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud system with TK cross + cloud filter from 1d timeframe
# Uses Ichimoku (Tenkan-sen, Kijun-sen, Senkou Span A/B) on daily timeframe
# to determine trend direction and dynamic support/resistance
# TK cross provides entry signals, cloud acts as dynamic filter
# Works in bull/bear by only taking trades in direction of cloud color
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = np.full(len(df_1d), np.nan)
    for i in range(tenkan_period - 1, len(df_1d)):
        window_high = np.max(high_1d[i - tenkan_period + 1:i + 1])
        window_low = np.min(low_1d[i - tenkan_period + 1:i + 1])
        tenkan_sen[i] = (window_high + window_low) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = np.full(len(df_1d), np.nan)
    for i in range(kijun_period - 1, len(df_1d)):
        window_high = np.max(high_1d[i - kijun_period + 1:i + 1])
        window_low = np.min(low_1d[i - kijun_period + 1:i + 1])
        kijun_sen[i] = (window_high + window_low) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_span_a = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26 periods
    senkou_span_b = np.full(len(df_1d), np.nan)
    for i in range(senkou_span_b_period - 1, len(df_1d)):
        window_high = np.max(high_1d[i - senkou_span_b_period + 1:i + 1])
        window_low = np.min(low_1d[i - senkou_span_b_period + 1:i + 1])
        senkou_span_b[i] = (window_high + window_low) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for 1d close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 26  # +26 for forward shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine cloud boundaries and color
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        bullish_cloud = senkou_a_aligned[i] > senkou_b_aligned[i]  # Green cloud
        bearish_cloud = senkou_a_aligned[i] < senkou_b_aligned[i]  # Red cloud
        
        # TK Cross signals
        tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud (in bullish cloud)
            if tk_cross_bullish and price > upper_cloud and bullish_cloud:
                signals[i] = size
                position = 1
            # Short: TK cross bearish AND price below cloud (in bearish cloud)
            elif tk_cross_bearish and price < lower_cloud and bearish_cloud:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if not tk_cross_bullish or price < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if not tk_cross_bearish or price > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_1dFilter"
timeframe = "6h"
leverage = 1.0