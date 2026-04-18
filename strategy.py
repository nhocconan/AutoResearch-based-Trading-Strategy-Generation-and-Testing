#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter
Hypothesis: Ichimoku cloud with TK cross and daily trend filter works across bull/bear markets.
- Uses 1d cloud (Senkou Span A/B) for trend filter and support/resistance
- TK cross (Tenkan/Kijun) on 6h for entry timing
- Only trade when price is above/below cloud with TK cross confirmation
- Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun_sen[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = np.full(n, np.nan)
    valid_idx = ~(np.isnan(tenkan_sen) | np.isnan(kijun_sen))
    senkou_span_a[valid_idx] = (tenkan_sen[valid_idx] + kijun_sen[valid_idx]) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        senkou_span_b[i] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Get 1-day data for cloud filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1-day Ichimoku components
    tenkan_1d = np.full(len(high_1d), np.nan)
    kijun_1d = np.full(len(high_1d), np.nan)
    senkou_span_a_1d = np.full(len(high_1d), np.nan)
    senkou_span_b_1d = np.full(len(high_1d), np.nan)
    
    for i in range(tenkan_period - 1, len(high_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(high_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    
    valid_1d = ~(np.isnan(tenkan_1d) | np.isnan(kijun_1d))
    senkou_span_a_1d[valid_1d] = (tenkan_1d[valid_1d] + kijun_1d[valid_1d]) / 2
    
    for i in range(senkou_span_b_period - 1, len(high_1d)):
        senkou_span_b_1d[i] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # Align 1-day Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Cloud boundaries from 1d (shifted forward by 26 periods)
    # Senkou Span A and B are plotted 26 periods ahead
    senkou_span_a_1d_shifted = np.full_like(senkou_span_a_1d_aligned, np.nan)
    senkou_span_b_1d_shifted = np.full_like(senkou_span_b_1d_aligned, np.nan)
    
    if len(senkou_span_a_1d_aligned) > 26:
        senkou_span_a_1d_shifted[26:] = senkou_span_a_1d_aligned[:-26]
        senkou_span_b_1d_shifted[26:] = senkou_span_b_1d_aligned[:-26]
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_1d_shifted, senkou_span_b_1d_shifted)
    cloud_bottom = np.minimum(senkou_span_a_1d_shifted, senkou_span_b_1d_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 26
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # TK cross signals
        tk_cross_up = tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        if position == 0:
            # Long: price above cloud + TK cross up + 1-day bullish (Tenkan > Kijun)
            if (close[i] > cloud_top[i] and tk_cross_up and 
                tenkan_1d_aligned[i] > kijun_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross down + 1-day bearish (Tenkan < Kijun)
            elif (close[i] < cloud_bottom[i] and tk_cross_down and 
                  tenkan_1d_aligned[i] < kijun_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below cloud or TK cross down
            if close[i] < cloud_bottom[i] or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or TK cross up
            if close[i] > cloud_top[i] or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter"
timeframe = "6h"
leverage = 1.0