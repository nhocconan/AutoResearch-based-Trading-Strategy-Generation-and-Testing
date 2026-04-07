#!/usr/bin/env python3
"""
6h_ichimoku_1d_trend_v1
Hypothesis: On 6h timeframe, enter long when Tenkan-sen crosses above Kijun-sen with price above Kumo (cloud) and 1d trend bullish (price above 1d Kumo), enter short when Tenkan-sen crosses below Kijun-sen with price below Kumo and 1d trend bearish. Uses 1w Ichimoku for higher timeframe trend filter. Designed for 15-35 trades/year to minimize fee drag while capturing trend continuation in both bull and bear markets by aligning with higher timeframe Ichimoku structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou_span = pd.Series(close)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Ichimoku
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # Calculate 1d Ichimoku for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Calculate 1w Ichimoku for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Align 1d and 1w Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after Ichimoku warmup period
        # Skip if data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or 
            np.isnan(senkou_b_6h[i]) or np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or np.isnan(senkou_a_1d_aligned[i]) or
            np.isnan(senkou_b_1d_aligned[i]) or np.isnan(tenkan_1w_aligned[i]) or
            np.isnan(kijun_1w_aligned[i]) or np.isnan(senkou_a_1w_aligned[i]) or
            np.isnan(senkou_b_1w_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Determine Kumo (cloud) boundaries for 6h
        upper_kumo_6h = max(senkou_a_6h[i], senkou_b_6h[i])
        lower_kumo_6h = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Determine Kumo (cloud) boundaries for 1d
        upper_kumo_1d = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_kumo_1d = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Determine Kumo (cloud) boundaries for 1w
        upper_kumo_1w = max(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        lower_kumo_1w = min(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        
        if position == 1:  # Long position
            # Exit: Tenkan-sen crosses below Kijun-sen OR price falls below 6h Kumo
            if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]) or close[i] < lower_kumo_6h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Tenkan-sen crosses above Kijun-sen OR price rises above 6h Kumo
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]) or close[i] > upper_kumo_6h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bullish conditions: Tenkan above Kijun, price above Kumo, aligned with higher timeframes
            bullish_cross = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
            price_above_kumo_6h = close[i] > upper_kumo_6h
            price_above_kumo_1d = close[i] > upper_kumo_1d
            price_above_kumo_1w = close[i] > upper_kumo_1w
            
            # Bearish conditions: Tenkan below Kijun, price below Kumo, aligned with higher timeframes
            bearish_cross = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
            price_below_kumo_6h = close[i] < lower_kumo_6h
            price_below_kumo_1d = close[i] < lower_kumo_1d
            price_below_kumo_1w = close[i] < lower_kumo_1w
            
            if bullish_cross and price_above_kumo_6h and price_above_kumo_1d and price_above_kumo_1w:
                position = 1
                signals[i] = 0.25
            elif bearish_cross and price_below_kumo_6h and price_below_kumo_1d and price_below_kumo_1w:
                position = -1
                signals[i] = -0.25
    
    return signals