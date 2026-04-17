#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_DailyFilter_V1
Hypothesis: On 6h timeframe, use daily Ichimoku cloud as trend filter and Tenkan/Kijun cross for entry.
Long when price above cloud and Tenkan > Kijun. Short when price below cloud and Tenkan < Kijun.
Ichimoku cloud from daily timeframe provides strong trend context that works in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year) with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === Daily Data (HTF for Ichimoku) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    
    # Warmup period - need enough data for Ichimoku calculations
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or
            np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above cloud AND Tenkan > Kijun (bullish cross)
            if close[i] > cloud_top and tenkan_1d_aligned[i] > kijun_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below cloud AND Tenkan < Kijun (bearish cross)
            elif close[i] < cloud_bottom and tenkan_1d_aligned[i] < kijun_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit when price crosses below cloud OR Tenkan < Kijun
            if close[i] < cloud_bottom or tenkan_1d_aligned[i] < kijun_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above cloud OR Tenkan > Kijun
            if close[i] > cloud_top or tenkan_1d_aligned[i] > kijun_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_DailyFilter_V1"
timeframe = "6h"
leverage = 1.0