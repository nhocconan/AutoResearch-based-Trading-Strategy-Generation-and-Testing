#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_v1
# Hypothesis: Ichimoku cloud (Tenkan-sen/Kijun-sen cross) with daily cloud filter on 6h chart.
# Long when: Tenkan > Kijun AND price > Kumo cloud (Senkou Span A/B) AND daily trend up (price > daily Kumo).
# Short when: Tenkan < Kijun AND price < Kumo cloud AND daily trend down (price < daily Kumo).
# Uses cloud as dynamic support/resistance and daily timeframe for trend filter.
# Works in both bull and bear markets due to cloud acting as dynamic S/R and trend filter reducing whipsaw.
# Target: 15-35 trades/year (60-140 total over 4 years) with strict entry conditions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period)
    def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
        # Tenkan-sen: (9-period high + 9-period low)/2
        tenkan_sen = np.full(n, np.nan)
        for i in range(tenkan-1, n):
            tenkan_sen[i] = (np.max(high[i-tenkan+1:i+1]) + np.min(low[i-tenkan+1:i+1])) / 2
        
        # Kijun-sen: (26-period high + 26-period low)/2
        kijun_sen = np.full(n, np.nan)
        for i in range(kijun-1, n):
            kijun_sen[i] = (np.max(high[i-kijun+1:i+1]) + np.min(low[i-kijun+1:i+1])) / 2
        
        # Senkou Span A: (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
        senkou_a = np.full(n, np.nan)
        for i in range(n):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                idx = i + kijun  # shifted ahead by kijun periods
                if idx < n:
                    senkou_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
        
        # Senkou Span B: (52-period high + 52-period low)/2 shifted 26 periods ahead
        senkou_b = np.full(n, np.nan)
        for i in range(n):
            if i >= senkou-1:
                idx = i + kijun  # shifted ahead by kijun periods
                if idx < n:
                    senkou_b[idx] = (np.max(high[i-senkou+1:i+1]) + np.min(low[i-senkou+1:i+1])) / 2
        
        return tenkan_sen, kijun_sen, senkou_a, senkou_b
    
    tenkan_sen, kijun_sen, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Kumo cloud boundaries: Senkou Span A and B
    # For cloud top/bottom calculation
    kumo_top = np.full(n, np.nan)
    kumo_bottom = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(senkou_a[i]) and not np.isnan(senkou_b[i]):
            kumo_top[i] = max(senkou_a[i], senkou_b[i])
            kumo_bottom[i] = min(senkou_a[i], senkou_b[i])
    
    # Load 1d data ONCE before loop for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Ichimoku cloud
    tenkan_sen_1d = np.full(len(close_1d), np.nan)
    kijun_sen_1d = np.full(len(close_1d), np.nan)
    senkou_a_1d = np.full(len(close_1d), np.nan)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    
    # Daily Tenkan-sen (9-period)
    for i in range(9-1, len(high_1d)):
        tenkan_sen_1d[i] = (np.max(high_1d[i-9+1:i+1]) + np.min(low_1d[i-9+1:i+1])) / 2
    
    # Daily Kijun-sen (26-period)
    for i in range(26-1, len(high_1d)):
        kijun_sen_1d[i] = (np.max(high_1d[i-26+1:i+1]) + np.min(low_1d[i-26+1:i+1])) / 2
    
    # Daily Senkou Span A
    for i in range(len(close_1d)):
        if not np.isnan(tenkan_sen_1d[i]) and not np.isnan(kijun_sen_1d[i]):
            idx = i + 26
            if idx < len(close_1d):
                senkou_a_1d[idx] = (tenkan_sen_1d[i] + kijun_sen_1d[i]) / 2
    
    # Daily Senkou Span B (52-period)
    for i in range(len(close_1d)):
        if i >= 52-1:
            idx = i + 26
            if idx < len(close_1d):
                senkou_b_1d[idx] = (np.max(high_1d[i-52+1:i+1]) + np.min(low_1d[i-52+1:i+1])) / 2
    
    # Daily Kumo cloud
    kumotop_1d = np.full(len(close_1d), np.nan)
    kumbottom_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(senkou_a_1d[i]) and not np.isnan(senkou_b_1d[i]):
            kumotop_1d[i] = max(senkou_a_1d[i], senkou_b_1d[i])
            kumbottom_1d[i] = min(senkou_a_1d[i], senkou_b_1d[i])
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    kumotop_1d_aligned = align_htf_to_ltf(prices, df_1d, kumotop_1d)
    kumbottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumbottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(kumotop_1d_aligned[i]) or np.isnan(kumbottom_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud relationship: price above/below cloud
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        
        # TK cross: Tenkan-sen vs Kijun-sen
        tk_cross_up = tenkan_sen[i] > kijun_sen[i]
        tk_cross_down = tenkan_sen[i] < kijun_sen[i]
        
        # Daily trend filter: price vs daily cloud
        daily_uptrend = close[i] > kumotop_1d_aligned[i]
        daily_downtrend = close[i] < kumbottom_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price falls below cloud OR TK cross down
            if price_below_kumo or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above cloud OR TK cross up
            if price_above_kumo or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud + TK cross up + daily uptrend
            if price_above_kumo and tk_cross_up and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud + TK cross down + daily downtrend
            elif price_below_kumo and tk_cross_down and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals