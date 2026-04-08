#1/2025-01-11
#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_breakout_v1
# Hypothesis: 6-hour Ichimoku cloud breakout with 1-day trend filter.
# Long when price breaks above Kumo (cloud) and Tenkan-sen > Kijun-sen, with price above 1-day Kumo.
# Short when price breaks below Kumo and Tenkan-sen < Kijun-sen, with price below 1-day Kumo.
# Exit when price re-enters Kumo or Tenkan/Kijun cross reverses.
# Uses Ichimoku on 6h for entry timing and 1d for trend filter to avoid counter-trend trades.
# Designed to generate ~15-30 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = np.full(n, np.nan)
    for i in range(period_tenkan - 1, n):
        tenkan_sen[i] = (np.max(high[i - period_tenkan + 1:i + 1]) + 
                         np.min(low[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = np.full(n, np.nan)
    for i in range(period_kijun - 1, n):
        kijun_sen[i] = (np.max(high[i - period_kijun + 1:i + 1]) + 
                        np.min(low[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + period_kijun  # Plot 26 periods ahead
            if idx < n:
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(n, np.nan)
    for i in range(period_senkou_b - 1, n):
        senkou_span_b[i] = (np.max(high[i - period_senkou_b + 1:i + 1]) + 
                            np.min(low[i - period_senkou_b + 1:i + 1])) / 2
    for i in range(n):
        if not np.isnan(senkou_span_b[i]):
            idx = i + period_kijun  # Plot 26 periods ahead
            if idx < n:
                senkou_span_b[idx] = senkou_span_b[i]
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Ichimoku on 1-day for trend filter (same periods)
    tenkan_sen_1d = np.full(len(high_1d), np.nan)
    for i in range(period_tenkan - 1, len(high_1d)):
        tenkan_sen_1d[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                            np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    kijun_sen_1d = np.full(len(high_1d), np.nan)
    for i in range(period_kijun - 1, len(high_1d)):
        kijun_sen_1d[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                           np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    senkou_span_a_1d = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(tenkan_sen_1d[i]) and not np.isnan(kijun_sen_1d[i]):
            idx = i + period_kijun
            if idx < len(high_1d):
                senkou_span_a_1d[idx] = (tenkan_sen_1d[i] + kijun_sen_1d[i]) / 2
    
    senkou_span_b_1d = np.full(len(high_1d), np.nan)
    for i in range(period_senkou_b - 1, len(high_1d)):
        senkou_span_b_1d[i] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                               np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    for i in range(len(high_1d)):
        if not np.isnan(senkou_span_b_1d[i]):
            idx = i + period_kijun
            if idx < len(high_1d):
                senkou_span_b_1d[idx] = senkou_span_b_1d[i]
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        
        # 1-day Ichimoku for trend filter
        tenkan_1d = tenkan_sen_1d_aligned[i]
        kijun_1d = kijun_sen_1d_aligned[i]
        span_a_1d = senkou_span_a_1d_aligned[i]
        span_b_1d = senkou_span_b_1d_aligned[i]
        
        # Kumo (cloud) boundaries
        upper_kumo = max(span_a, span_b)
        lower_kumo = min(span_a, span_b)
        upper_kumo_1d = max(span_a_1d, span_b_1d)
        lower_kumo_1d = min(span_a_1d, span_b_1d)
        
        if position == 1:  # Long
            # Exit: price re-enters Kumo or Tenkan/Kijun cross reverses (Tenkan < Kijun)
            if price <= upper_kumo and price >= lower_kumo:
                position = 0
                signals[i] = 0.0
            elif tenkan < kijun:  # Bearish cross
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price re-enters Kumo or Tenkan/Kijun cross reverses (Tenkan > Kijun)
            if price <= upper_kumo and price >= lower_kumo:
                position = 0
                signals[i] = 0.0
            elif tenkan > kijun:  # Bullish cross
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Kumo breakout with alignment
            # Bullish: price breaks above Kumo, Tenkan > Kijun, and price above 1d Kumo
            if price > upper_kumo and tenkan > kijun and price > upper_kumo_1d:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below Kumo, Tenkan < Kijun, and price below 1d Kumo
            elif price < lower_kumo and tenkan < kijun and price < lower_kumo_1d:
                position = -1
                signals[i] = -0.25
    
    return signals