#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Hypothesis: Ichimoku Kinko Hyo on 1d timeframe provides strong trend direction and support/resistance levels. 
6h timeframe entry when price crosses above/below Kumo (cloud) with TK cross confirmation, 
filtered by weekly trend to avoid counter-trend trades. Ichimoku works well in both trending and ranging markets,
and the multi-timeframe alignment reduces false signals. Target: 60-120 trades over 4 years (15-30/year).
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full(len(high_1d), np.nan)
    if len(high_1d) >= period_tenkan:
        for i in range(period_tenkan - 1, len(high_1d)):
            tenkan_sen[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                           np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full(len(high_1d), np.nan)
    if len(high_1d) >= period_kijun:
        for i in range(period_kijun - 1, len(high_1d)):
            kijun_sen[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                          np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full(len(high_1d), np.nan)
    if len(high_1d) >= period_kijun + period_tenkan:
        for i in range(len(high_1d)):
            idx = i + period_kijun  # shift forward
            if idx < len(high_1d) and not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(len(high_1d), np.nan)
    if len(high_1d) >= period_senkou_b:
        for i in range(period_senkou_b - 1, len(high_1d)):
            idx = i + period_kijun  # shift forward
            if idx < len(high_1d):
                senkou_span_b[idx] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                                    np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = close_1w[i] * alpha + ema20_1w[i-1] * (1 - alpha)
    
    # Align weekly EMA20 to 6h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52 + 26, 20)  # Ensure Ichimoku and volume ready
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries and color
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_green = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]  # bullish cloud
        
        if position == 0:
            # Long: price above cloud, TK cross bullish, weekly uptrend
            if (close[i] > upper_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and  # TK cross bullish
                close[i] > ema20_1w_aligned[i] and  # weekly uptrend
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, TK cross bearish, weekly downtrend
            elif (close[i] < lower_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and  # TK cross bearish
                  close[i] < ema20_1w_aligned[i] and  # weekly downtrend
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below cloud or TK cross turns bearish
            if (close[i] < lower_cloud or 
                tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or TK cross turns bullish
            if (close[i] > upper_cloud or 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0