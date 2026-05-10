#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_1wFilter_Volume
# Hypothesis: 6-hour Ichimoku cloud strategy with weekly trend filter and volume confirmation.
# In bull markets: price above weekly Kumo cloud with TK cross bullish and volume spike = long.
# In bear markets: price below weekly Kumo cloud with TK cross bearish and volume spike = short.
# Weekly trend filter avoids counter-trend trades; Ichimoku cloud acts as dynamic support/resistance;
# Volume confirmation ensures breakout strength. Designed for 6h to achieve 12-37 trades/year.

name = "6h_Ichimoku_Cloud_Trend_1wFilter_Volume"
timeframe = "6h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Weekly data for trend filter and Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    def calculate_ichimoku(h, l, c):
        tenkan_sen = (np.max(h[-9:]) + np.min(l[-9:])) / 2 if len(h) >= 9 else np.nan
        kijun_sen = (np.max(h[-26:]) + np.min(l[-26:])) / 2 if len(h) >= 26 else np.nan
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2) if not (np.isnan(tenkan_sen) or np.isnan(kijun_sen)) else np.nan
        senkou_span_b = (np.max(h[-52:]) + np.min(l[-52:])) / 2 if len(h) >= 52 else np.nan
        chikou_span = c[-1] if len(c) >= 1 else np.nan
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    # Calculate Ichimoku for each weekly bar (using expanding window)
    tenkan = np.full_like(close_1w, np.nan)
    kijun = np.full_like(close_1w, np.nan)
    senkou_a = np.full_like(close_1w, np.nan)
    senkou_b = np.full_like(close_1w, np.nan)
    chikou = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 52:  # Need full 52 periods for Senkou B
            h_slice = high_1w[max(0, i-51):i+1]
            l_slice = low_1w[max(0, i-51):i+1]
            c_slice = close_1w[max(0, i-51):i+1]
            tenkan[i], kijun[i], senkou_a[i], senkou_b[i], chikou[i] = calculate_ichimoku(h_slice, l_slice, c_slice)
        elif i >= 26:  # Need at least 26 periods for Kijun
            h_slice = high_1w[max(0, i-25):i+1]
            l_slice = low_1w[max(0, i-25):i+1]
            c_slice = close_1w[max(0, i-25):i+1]
            tenkan[i], kijun[i], senkou_a[i], senkou_b[i], chikou[i] = calculate_ichimoku(h_slice, l_slice, c_slice)
        elif i >= 9:  # Need at least 9 periods for Tenkan
            h_slice = high_1w[max(0, i-8):i+1]
            l_slice = low_1w[max(0, i-8):i+1]
            c_slice = close_1w[max(0, i-8):i+1]
            tenkan[i], kijun[i], senkou_a[i], senkou_b[i], chikou[i] = calculate_ichimoku(h_slice, l_slice, c_slice)
    
    # Kumo cloud boundaries (Senkou Span A and B shifted forward by 26 periods)
    # For cloud at time t, we use Senkou Span values from t-26
    senkou_a_shifted = np.full_like(senkou_a, np.nan)
    senkou_b_shifted = np.full_like(senkou_b, np.nan)
    for i in range(26, len(senkou_a)):
        senkou_a_shifted[i] = senkou_a[i-26]
        senkou_b_shifted[i] = senkou_b[i-26]
    
    # Kumo cloud: upper band = max(Senkou A, Senkou B), lower band = min(Senkou A, Senkou B)
    kumo_top = np.full_like(senkou_a_shifted, np.nan)
    kumo_bottom = np.full_like(senkou_a_shifted, np.nan)
    for i in range(len(kumo_top)):
        if not (np.isnan(senkou_a_shifted[i]) or np.isnan(senkou_b_shifted[i])):
            kumo_top[i] = max(senkou_a_shifted[i], senkou_b_shifted[i])
            kumo_bottom[i] = min(senkou_a_shifted[i], senkou_b_shifted[i])
    
    # Weekly volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1w, 20)
    
    # Align weekly indicators to 6h timeframe (wait for weekly bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1w, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above Kumo cloud, TK cross bullish, strong volume
            if (close[i] > kumo_top_aligned[i] and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                volume[i] > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below Kumo cloud, TK cross bearish, strong volume
            elif (close[i] < kumo_bottom_aligned[i] and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Kumo cloud or TK cross turns bearish
            if close[i] < kumo_bottom_aligned[i] or tenkan_aligned[i] < kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Kumo cloud or TK cross turns bullish
            if close[i] > kumo_top_aligned[i] or tenkan_aligned[i] > kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals