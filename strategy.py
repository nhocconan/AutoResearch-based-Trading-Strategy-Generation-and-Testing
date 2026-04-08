#!/usr/bin/env python3
# 6h_ichimoku_cloud_1d_filter_v1
# Hypothesis: Uses Ichimoku Cloud from daily timeframe for trend filter, with Tenkan-Kijun cross on 6h for entry.
# In bull markets: price above cloud + TK cross up = long.
# In bear markets: price below cloud + TK cross down = short.
# Cloud acts as dynamic support/resistance, reducing whipsaws. TK cross provides timely entries.
# Designed for ~20-40 trades/year on 6h to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_1d_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i < tenkan_period - 1:
            tenkan_sen[i] = np.nan
        else:
            period_high = np.max(high_1d[i-tenkan_period+1:i+1])
            period_low = np.min(low_1d[i-tenkan_period+1:i+1])
            tenkan_sen[i] = (period_high + period_low) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i < kijun_period - 1:
            kijun_sen[i] = np.nan
        else:
            period_high = np.max(high_1d[i-kijun_period+1:i+1])
            period_low = np.min(low_1d[i-kijun_period+1:i+1])
            kijun_sen[i] = (period_high + period_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
    senkou_span_a = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        idx = i + kijun_period  # 26 periods ahead
        if idx < len(close_1d) and not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
        else:
            senkou_span_a[i] = np.nan
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods, plotted 26 periods ahead
    senkou_span_b = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i < senkou_span_b_period - 1:
            senkou_span_b[i] = np.nan
        else:
            period_high = np.max(high_1d[i-senkou_span_b_period+1:i+1])
            period_low = np.min(low_1d[i-senkou_span_b_period+1:i+1])
            senkou_span_b[i] = (period_high + period_low) / 2
    # Shift Senkou Span B 26 periods ahead
    senkou_span_b_shifted = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        idx = i + kijun_period  # 26 periods ahead
        if idx < len(close_1d) and not np.isnan(senkou_span_b[i]):
            senkou_span_b_shifted[idx] = senkou_span_b[i]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 52  # Ensure Ichimoku is ready
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Calculate cloud boundaries (Senkou Span A and B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK cross signals
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: price below cloud or TK cross down
            if close[i] < cloud_bottom or tk_cross_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above cloud or TK cross up
            if close[i] > cloud_top or tk_cross_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above cloud and TK cross up
            if close[i] > cloud_top and tk_cross_up:
                position = 1
                signals[i] = 0.25
            # Short entry: price below cloud and TK cross down
            elif close[i] < cloud_bottom and tk_cross_down:
                position = -1
                signals[i] = -0.25
    
    return signals