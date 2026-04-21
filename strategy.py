#!/usr/bin/env python3
"""
6h_IchimokuTK_Cross_CloudFilter_V1
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with cloud filter from 1d timeframe. 
Long when TK crosses above AND price > 1d cloud top (senkou span A/B max). 
Short when TK crosses below AND price < 1d cloud bottom (min). 
Uses weekly HTF for regime: only trade in direction of weekly trend (price > weekly EMA50 for longs, < for shorts).
Designed for low trade frequency (50-150 total 6h trades over 4 years) to minimize fee drag and capture trends in both bull/bear markets via multi-timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Ichimoku, 1w for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Ichimoku Cloud (9, 26, 52) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (9-period): (highest high + lowest low)/2 over 9 periods
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    
    # Kijun-sen (26-period): (highest high + lowest low)/2 over 26 periods
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    
    # Senkou Span A (leading span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (leading span B): (highest high + lowest low)/2 over 52 periods plotted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) 
            or np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])
            or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Cloud boundaries (top = max(span A, B), bottom = min(span A, B))
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Weekly trend filter
        weekly_uptrend = price > ema_50_1w_aligned[i]
        weekly_downtrend = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: TK cross above + price above cloud + weekly uptrend
            if tenkan > kijun and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and price > cloud_top and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below + price below cloud + weekly downtrend
            elif tenkan < kijun and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and price < cloud_bottom and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross below OR price drops below cloud bottom
            if tenkan < kijun or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross above OR price rises above cloud top
            if tenkan > kijun or price > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchimokuTK_Cross_CloudFilter_V1"
timeframe = "6h"
leverage = 1.0