#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_Volume
Hypothesis: Ichimoku conversion/base line cross with cloud filter from 1d timeframe.
Long when TK cross above cloud in uptrend, short when TK cross below cloud in downtrend.
Uses 1d Ichimoku for higher timeframe trend/filter and 6h for entry timing.
Targets 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.
Works in bull via trend continuation and bear via counter-trend pulls to cloud edges.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_sen = np.full(len(close_1d), np.nan)
    for i in range(period_tenkan - 1, len(close_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_sen = np.full(len(close_1d), np.nan)
    for i in range(period_kijun - 1, len(close_1d)):
        kijun_sen[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_span_b = np.full(len(close_1d), np.nan)
    for i in range(period_senkou_b - 1, len(close_1d)):
        senkou_span_b[i] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Ichimoku (52 periods) and volume MA (20)
    start_idx = max(52, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(span_a_aligned[i]) or np.isnan(span_b_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(span_a_aligned[i], span_b_aligned[i])
        lower_cloud = np.minimum(span_a_aligned[i], span_b_aligned[i])
        
        # TK Cross signals
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = price > upper_cloud
        price_below_cloud = price < lower_cloud
        price_in_cloud = (price >= lower_cloud) & (price <= upper_cloud)
        
        # Trend filter: price above/below cloud
        uptrend = price_above_cloud
        downtrend = price_below_cloud
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: TK cross above + price above cloud in uptrend with volume
            if tk_cross_above and price_above_cloud and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: TK cross below + price below cloud in downtrend with volume
            elif tk_cross_below and price_below_cloud and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TK cross below or price drops into/below cloud
            if tk_cross_below or price_in_cloud or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: TK cross above or price rises into/above cloud
            if tk_cross_above or price_in_cloud or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0