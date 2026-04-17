#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Tenkan_Kijun_Cross_v1
6-hour strategy using Ichimoku cloud with daily trend filter.
- Tenkan/Kijun cross on 6h timeframe for entry signals
- Cloud color (Senkou Span A/B) from daily timeframe as trend filter
- Only take longs when price above daily cloud, shorts when below
- Uses 6-period Tenkan, 22-period Kijun, 44-period Senkou
- Designed to work in both bull and bear markets by following daily trend
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
    
    # === Calculate Daily Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b_1d = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align daily Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # === Calculate 6h Tenkan and Kijun for Cross Signals ===
    period_tenkan_6h = 9
    period_kijun_6h = 26
    
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # Calculate cross signals
    tenkan_prev = np.roll(tenkan_6h, 1)
    kijun_prev = np.roll(kijun_6h, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    # Bullish cross: Tenkan crosses above Kijun
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_prev <= kijun_prev)
    # Bearish cross: Tenkan crosses below Kijun
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_prev >= kijun_prev)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine cloud color and price position relative to cloud
        # Green cloud: Senkou Span A > Senkou Span B (bullish)
        # Red cloud: Senkou Span A < Senkou Span B (bearish)
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        cloud_red = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Price above/below cloud
        price_above_cloud = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish TK cross + price above green cloud
            if tk_cross_up[i] and cloud_green and price_above_cloud:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish TK cross + price below red cloud
            elif tk_cross_down[i] and cloud_red and price_below_cloud:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: bearish TK cross OR price falls below cloud
            if tk_cross_down[i] or not price_above_cloud:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish TK cross OR price rises above cloud
            if tk_cross_up[i] or not price_below_cloud:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Tenkan_Kijun_Cross_v1"
timeframe = "6h"
leverage = 1.0