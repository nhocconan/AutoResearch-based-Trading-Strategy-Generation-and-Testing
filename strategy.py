#!/usr/bin/env python3
"""
6h_1w_Ichimoku_TK_Cross_CloudFilter
Hypothesis: Ichimoku Tenkan/Kijun cross with cloud filter from weekly timeframe provides high-probability trend signals in both bull and bear markets. Using weekly cloud as macro trend filter reduces whipsaws, while TK cross on 6h captures intermediate swings. Designed for low trade frequency (15-35/year) with 0.25 position size to manage drawdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for Ichimoku cloud
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:  # Need ~1 year of weekly data
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_weekly).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_weekly).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_weekly).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_weekly).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(1)  # Shifted for cloud
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high_weekly).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                      pd.Series(low_weekly).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2).shift(1)
    
    # Align weekly Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_weekly, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_weekly, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_weekly, senkou_span_b.values)
    
    # Main timeframe data (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 1.5 * vol_ma
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud
            if tenkan > kijun and price > cloud_top and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud
            elif tenkan < kijun and price < cloud_bottom and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud bottom
            if tenkan < kijun or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud top
            if tenkan > kijun or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0