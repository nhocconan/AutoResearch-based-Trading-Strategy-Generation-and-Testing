#!/usr/bin/env python3
"""
6h_Ichimoku_TKCross_1dTrendFilter
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with daily trend filter (price above/below Kumo cloud) provides high-probability entries. Works in bull (TK cross up + price above cloud) and bear (TK cross down + price below cloud). Reduces false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_52 + min_low_52) / 2)
    
    # Align Ichimoku components to 6h (with proper look-ahead prevention)
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo cloud boundaries (Senkou Span A and B)
    # Senkou Span A and B are already shifted ahead in their calculation
    # For cloud filter, we need current cloud (values from 26 periods ago)
    # But since we aligned the already-shifted Senkou spans, we use them directly
    # The cloud at time t is formed by Senkou A and B from 26 periods ago
    # So we need to shift the aligned Senkou spans BACK by 26 periods to get current cloud
    # However, align_htf_to_ltf already handles the shift correctly for Ichimoku
    # The Senkou spans we calculated are already future-shifted, so when aligned,
    # they represent the cloud that should be visible NOW
    
    # Kumo cloud top and bottom
    kumo_top = np.maximum(senkou_a_6h, senkou_b_6h)
    kumo_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h <= kijun_6h)  # Crossed up
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h >= kijun_6h)  # Crossed down
    
    # Price relative to cloud
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross up + price above Kumo cloud
            if tk_cross_up[i] and price_above_kumo[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down + price below Kumo cloud
            elif tk_cross_down[i] and price_below_kumo[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down OR price falls below Kumo cloud
            if tk_cross_down[i] or close[i] < kumo_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up OR price rises above Kumo cloud
            if tk_cross_up[i] or close[i] > kumo_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_1dTrendFilter"
timeframe = "6h"
leverage = 1.0