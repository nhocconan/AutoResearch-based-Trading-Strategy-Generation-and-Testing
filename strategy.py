#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_1dTrend
# Hypothesis: Use Ichimoku cloud from 1d timeframe for trend direction and support/resistance.
# Enter long when price breaks above the Kumo (cloud) top and price > Kijun (base line).
# Enter short when price breaks below the Kumo (cloud) bottom and price < Kijun.
# Uses weekly trend filter (price > weekly EMA50) to avoid counter-trend trades.
# Designed for 15-30 trades/year on 6h timeframe with strong trend following edge.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            tenkan_sen[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1:
            kijun_sen[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
    senkou_span_a = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1 and i + period_kijun < len(high_1d):
            senkou_span_a[i + period_kijun] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, plotted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_senkou_b - 1 and i + period_kijun < len(high_1d):
            senkou_span_b[i + period_kijun] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(50)
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 48) / 50
    
    # Align weekly EMA to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Ichimoku calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kijun_sen_aligned[i]) or np.isnan(senkou_span_a_aligned[i]) or \
           np.isnan(senkou_span_b_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Enter long: Price breaks above cloud AND price > Kijun AND bullish weekly trend
            if close[i] > cloud_top and close[i] > kijun_sen_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below cloud AND price < Kijun AND bearish weekly trend
            elif close[i] < cloud_bottom and close[i] < kijun_sen_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below cloud or trend turns bearish
            if close[i] < cloud_bottom or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above cloud or trend turns bullish
            if close[i] > cloud_top or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals