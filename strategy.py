#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Twist_1dTrend
# Hypothesis: Use Ichimoku cloud twist (Tenkan/Kijun cross) on 6-timeframe for entry signals, filtered by 1-day trend (price above/below Kumo cloud).
# The cloud acts as dynamic support/resistance, reducing whipsaws in sideways markets.
# Trend filter ensures alignment with higher-timeframe momentum, improving signal quality in both bull and bear markets.
# Target: 15-35 trades/year to stay within optimal trade frequency for 6h.

name = "6h_Ichimoku_Cloud_Twist_1dTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 over last 9 periods
    def rolling_max(arr, window):
        return np.maximum.accumulate(np.where(np.arange(len(arr)) < window-1, np.nan, arr))[window-1:] if len(arr) >= window else np.full_like(arr, np.nan)
    
    def rolling_min(arr, window):
        return np.minimum.accumulate(np.where(np.arange(len(arr)) < window-1, np.nan, arr))[window-1:] if len(arr) >= window else np.full_like(arr, np.nan)
    
    # Calculate Tenkan-sen
    max_tenkan = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    min_tenkan = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (max_tenkan + min_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line)
    max_kijun = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    min_kijun = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (max_kijun + min_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B)
    max_senkou_b = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    min_senkou_b = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (max_senkou_b + min_senkou_b) / 2
    
    # Chikou Span (Lagging Span) - not used for signals but calculated for completeness
    # chikou_span = np.roll(close, -kijun_period)  # Not used in signal logic
    
    # 1-day trend filter: price relative to Ichimoku cloud from 1-day timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < senkou_span_b_period:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    max_tenkan_1d = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    min_tenkan_1d = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen_1d = (max_tenkan_1d + min_tenkan_1d) / 2
    
    max_kijun_1d = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    min_kijun_1d = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen_1d = (max_kijun_1d + min_kijun_1d) / 2
    
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    
    max_senkou_b_1d = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    min_senkou_b_1d = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1d = (max_senkou_b_1d + min_senkou_b_1d) / 2
    
    # Align 1d Ichimoku components to 6h timeframe
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Determine trend based on 1d cloud: price above cloud = uptrend, below cloud = downtrend
    # Cloud top = max(senkou_span_a, senkou_span_b), cloud bottom = min(senkou_span_a, senkou_span_b)
    cloud_top_1d = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun (bullish twist) AND price above 1d cloud AND volume confirmation
            if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and
                close[i] > cloud_top_1d[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun (bearish twist) AND price below 1d cloud AND volume confirmation
            elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
                  close[i] < cloud_bottom_1d[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Tenkan crosses below Kijun (bearish twist) OR price breaks below 1d cloud
            if (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1]) or \
               close[i] < cloud_bottom_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Tenkan crosses above Kijun (bullish twist) OR price breaks above 1d cloud
            if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1]) or \
               close[i] > cloud_top_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals