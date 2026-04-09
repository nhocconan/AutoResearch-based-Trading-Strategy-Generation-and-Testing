#!/usr/bin/env python3
# 6h_1d_ichimoku_trend_v1
# Hypothesis: 6-hour trend following using Ichimoku cloud from daily timeframe.
# Long when price > Senkou Span A/B and Tenkan > Kijun, short when opposite.
# Uses Kumo (cloud) as dynamic support/resistance and TK cross for momentum.
# Designed for 6h timeframe to capture medium-term trends with controlled trade frequency (target: 15-35/year).
# Works in bull markets (uptrend above cloud) and bear markets (downtrend below cloud).
# Uses daily Ichimoku to avoid noise, avoiding look-ahead bias via mtf_data helpers.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            tenkan_sen[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1:
            kijun_sen[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + period_kijun
            if idx < len(senkou_span_a):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= period_senkou_b - 1:
            idx = i + period_kijun
            if idx < len(senkou_span_b):
                senkou_span_b[idx] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup (need 52 days for Senkou B)
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud or TK cross turns bearish
            if close[i] < cloud_bottom or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud or TK cross turns bullish
            if close[i] > cloud_top or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud and bullish TK cross
            if close[i] > cloud_top and tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud and bearish TK cross
            elif close[i] < cloud_bottom and tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals