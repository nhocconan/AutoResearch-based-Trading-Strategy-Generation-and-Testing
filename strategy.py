#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_WeeklyTrend
Hypothesis: Ichimoku cloud from daily timeframe with weekly trend filter.
- Long when price breaks above Kumo (cloud) and weekly trend is up (weekly close > weekly EMA20)
- Short when price breaks below Kumo and weekly trend is down (weekly close < weekly EMA20)
- Exit when price returns to middle of Kumo (Kijun-sen) or weekly trend fails
- Uses weekly EMA20 for trend filter to avoid whipsaws in ranging markets
- Designed to capture strong trends while avoiding false breakouts in chop
- Target: 12-30 trades/year (48-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (using prior daily data to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan_sen = np.full(len(close_1d), np.nan)
    for i in range(period_tenkan - 1, len(close_1d)):
        tenkan_sen[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                         np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun_sen = np.full(len(close_1d), np.nan)
    for i in range(period_kijun - 1, len(close_1d)):
        kijun_sen[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                        np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full(len(close_1d), np.nan)
    for i in range(len(tenkan_sen)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + period_kijun
            if idx < len(senkou_span_a):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(len(close_1d), np.nan)
    for i in range(period_senkou_b - 1, len(close_1d)):
        idx = i + period_kijun
        if idx < len(senkou_span_b):
            senkou_span_b[idx] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                                  np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(period_senkou_b + period_kijun, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Calculate cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long entry: price breaks above cloud and weekly trend is up
            if (close[i] > upper_cloud and close_1d[-1] > ema20_1w_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below cloud and weekly trend is down
            elif (close[i] < lower_cloud and close_1d[-1] < ema20_1w_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle of cloud (Kijun-sen) or weekly trend fails
            if (close[i] <= kijun_sen_aligned[i] or close_1d[-1] < ema20_1w_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle of cloud (Kijun-sen) or weekly trend fails
            if (close[i] >= kijun_sen_aligned[i] or close_1d[-1] > ema20_1w_aligned[i] if len(close_1d) > 0 else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_WeeklyTrend"
timeframe = "6h"
leverage = 1.0