#!/usr/bin/env python3
"""
6h_IchiKumo_1dCloud_TKCross
Hypothesis: Trade Ichimoku Tenkan/Kijun cross with daily cloud filter on 6h timeframe.
Long when TK crosses above AND price above daily Kumo cloud; short when TK crosses below AND price below daily Kumo.
Uses daily Ichimoku for trend filter to avoid counter-trend trades in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year). Position size 0.25.
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
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = np.full_like(high_1d, np.nan)
    if len(high_1d) >= period_tenkan:
        for i in range(period_tenkan - 1, len(high_1d)):
            tenkan_sen[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                            np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = np.full_like(high_1d, np.nan)
    if len(high_1d) >= period_kijun:
        for i in range(period_kijun - 1, len(high_1d)):
            kijun_sen[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                           np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = np.full_like(high_1d, np.nan)
    valid_tenkan = ~np.isnan(tenkan_sen)
    valid_kijun = ~np.isnan(kijun_sen)
    valid_both = valid_tenkan & valid_kijun
    senkou_span_a[valid_both] = (tenkan_sen[valid_both] + kijun_sen[valid_both]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = np.full_like(high_1d, np.nan)
    if len(high_1d) >= period_senkou_b:
        for i in range(period_senkou_b - 1, len(high_1d)):
            senkou_span_b[i] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                               np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate Kumo cloud edges (Senkou Span A and B)
    # Cloud top = max(Senkou A, Senkou B)
    # Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26)  # Need enough data for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        # TK crossover signals
        tk_cross_above = (tenkan_sen_6h[i] > kijun_sen_6h[i] and 
                         tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1])
        tk_cross_below = (tenkan_sen_6h[i] < kijun_sen_6h[i] and 
                         tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1])
        
        if position == 0:
            # Long: TK cross above AND price above cloud
            if tk_cross_above and close[i] > cloud_top[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross below AND price below cloud
            elif tk_cross_below and close[i] < cloud_bottom[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TK cross below OR price below cloud
            if tk_cross_below or close[i] < cloud_bottom[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TK cross above OR price above cloud
            if tk_cross_above or close[i] > cloud_top[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_IchiKumo_1dCloud_TKCross"
timeframe = "6h"
leverage = 1.0