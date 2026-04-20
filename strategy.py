#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Concept: Ichimoku Cloud system on 6h timeframe with 1d filter.
- Long: TK Cross bullish + price above 1d Ichimoku cloud
- Short: TK Cross bearish + price below 1d Ichimoku cloud
- Exit: TK Cross reverses
- Position sizing: 0.25
- Target: 15-30 trades/year (60-120 total over 4 years)
- Works in bull/bear: Ichimoku adapts to volatility, cloud acts as dynamic support/resistance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
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
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # === 6h: Ichimoku TK Cross ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tenkan_sen, kijun_sen, _, _ = calculate_ichimoku(high, low, close, 9, 26, 52)
    
    # === Daily: Ichimoku Cloud (Senkou Span A & B) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    _, _, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d, 9, 26, 52)
    
    # Align daily cloud to 6h timeframe (Senkou Span A & B are already shifted)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Get values
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        senkou_a = senkou_a_aligned[i]
        senkou_b = senkou_b_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(senkou_a) or np.isnan(senkou_b)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if position == 0:
            # Long: TK Cross bullish + price above cloud
            if tenkan > kijun and close[i] > cloud_top:
                signals[i] = 0.25
                position = 1
            # Short: TK Cross bearish + price below cloud
            elif tenkan < kijun and close[i] < cloud_bottom:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK Cross turns bearish
            if tenkan < kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK Cross turns bullish
            if tenkan > kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals