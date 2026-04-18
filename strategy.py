#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross + Volume Confirmation
Hypothesis: Ichimoku provides a comprehensive trend system (cloud = support/resistance, TK cross = momentum). 
Using 1d Ichimoku for trend filter ensures alignment with higher timeframe momentum. 
TK cross on 6h with volume confirmation captures momentum shifts while avoiding false signals.
Works in bull markets (breakouts above cloud) and bear markets (breakdowns below cloud).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods shifted 26
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): Close shifted -22 periods (not used for signals)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Calculate 1d Ichimoku for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend: price above/below cloud
        # Cloud top = max(senkou_a, senkou_b), Cloud bottom = min(senkou_a, senkou_b)
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # 1d trend: 1 = uptrend (price above cloud), -1 = downtrend (price below cloud), 0 = in cloud
        if close_1d[i] > cloud_top_1d:
            trend_1d = 1
        elif close_1d[i] < cloud_bottom_1d:
            trend_1d = -1
        else:
            trend_1d = 0  # In cloud - no clear trend
        
        # 6h signals: TK cross and price relative to cloud
        tk_cross = tenkan[i] - kijun[i]
        tk_cross_prev = tenkan[i-1] - kijun[i-1] if i > 0 else 0
        
        # Cloud top/bottom for 6h
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        if position == 0:
            # Enter long: TK cross bullish + price above cloud + volume + 1d uptrend
            if (tk_cross > 0 and tk_cross_prev <= 0 and  # Bullish TK cross
                close[i] > cloud_top and                  # Price above cloud
                vol_confirm[i] and                        # Volume confirmation
                trend_1d == 1):                           # 1d uptrend
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross bearish + price below cloud + volume + 1d downtrend
            elif (tk_cross < 0 and tk_cross_prev >= 0 and   # Bearish TK cross
                  close[i] < cloud_bottom and               # Price below cloud
                  vol_confirm[i] and                        # Volume confirmation
                  trend_1d == -1):                          # 1d downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price falls below cloud
            if tk_cross < 0 or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if tk_cross > 0 or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_Volume_1dTrendFilter"
timeframe = "6h"
leverage = 1.0