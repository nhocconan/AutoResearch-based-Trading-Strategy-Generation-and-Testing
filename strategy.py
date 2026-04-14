#!/usr/bin/env python3
"""
6h Weekly Ichimoku Cloud Filter + Daily Pivot Reversal
Long when price is above weekly Ichimoku cloud and bounces from daily S1/S2 pivot.
Short when price is below weekly Ichimoku cloud and bounces from daily R1/R2 pivot.
Exit when price crosses the opposite pivot level or cloud is breached.
Designed for low turnover: ~15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data once for Ichimoku cloud
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 52:
        return np.zeros(n)
    
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Ichimoku cloud (9, 26, 52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_w).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_w).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_w).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_w).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b = senkou_span_b.shift(kijun_period)
    
    # Daily pivot points (using previous day)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Previous day's pivot levels
    pivot = (high_d + low_d + close_d) / 3
    r1 = 2 * pivot - low_d
    s1 = 2 * pivot - high_d
    r2 = pivot + (high_d - low_d)
    s2 = pivot - (high_d - low_d)
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_w, senkou_span_b.values)
    
    # Align daily pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_d, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(52, n):
        # Get current Ichimoku values
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Cloud boundaries (using Senkou Span A and B)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Get current daily pivot levels
        piv = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        
        if position == 0:
            # Long: Price above cloud AND bouncing from S1 or S2
            if close[i] > cloud_top and (abs(close[i] - s1_level) < 0.005 * close[i] or abs(close[i] - s2_level) < 0.005 * close[i]):
                position = 1
                signals[i] = position_size
            # Short: Price below cloud AND bouncing from R1 or R2
            elif close[i] < cloud_bottom and (abs(close[i] - r1_level) < 0.005 * close[i] or abs(close[i] - r2_level) < 0.005 * close[i]):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Price crosses below cloud OR breaks below S1
            if close[i] < cloud_bottom or close[i] < s1_level:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Price crosses above cloud OR breaks above R1
            if close[i] > cloud_top or close[i] > r1_level:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_WeeklyIchimoku_DailyPivot"
timeframe = "6h"
leverage = 1.0