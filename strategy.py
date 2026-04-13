#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1w Ichimoku Cloud and Tenkan/Kijun cross.
# Long: Price above Kumo cloud + Tenkan > Kijun (bullish cross)
# Short: Price below Kumo cloud + Tenkan < Kijun (bearish cross)
# Uses 1w Ichimoku for trend regime, 6h for execution.
# Ichimoku works in both bull/bear via cloud filter; cross signals capture momentum within trend.
# Target: 60-150 total trades over 4 years (15-38/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full(len(close_1w), np.nan)
    for i in range(tenkan_period - 1, len(close_1w)):
        period_high = np.max(high_1w[i - tenkan_period + 1:i + 1])
        period_low = np.min(low_1w[i - tenkan_period + 1:i + 1])
        tenkan_sen[i] = (period_high + period_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full(len(close_1w), np.nan)
    for i in range(kijun_period - 1, len(close_1w)):
        period_high = np.max(high_1w[i - kijun_period + 1:i + 1])
        period_low = np.min(low_1w[i - kijun_period + 1:i + 1])
        kijun_sen[i] = (period_high + period_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + kijun_period
            if idx < len(close_1w):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_span_b = np.full(len(close_1w), np.nan)
    for i in range(senkou_span_b_period - 1, len(close_1w)):
        period_high = np.max(high_1w[i - senkou_span_b_period + 1:i + 1])
        period_low = np.min(low_1w[i - senkou_span_b_period + 1:i + 1])
        senkou_span_b[i] = (period_high + period_low) / 2
        idx = i + kijun_period
        if idx < len(close_1w):
            senkou_span_b[idx] = senkou_span_b[i]
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Kumo cloud boundaries
        upper_cloud = max(span_a, span_b)
        lower_cloud = min(span_a, span_b)
        
        # Bullish conditions: price above cloud + Tenkan > Kijun
        bullish = (price > upper_cloud) and (tenkan > kijun)
        # Bearish conditions: price below cloud + Tenkan < Kijun
        bearish = (price < lower_cloud) and (tenkan < kijun)
        
        if position == 0:
            if bullish:
                position = 1
                signals[i] = position_size
            elif bearish:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below cloud OR Tenkan < Kijun (trend weakening)
            if (price < lower_cloud) or (tenkan < kijun):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above cloud OR Tenkan > Kijun (trend weakening)
            if (price > upper_cloud) or (tenkan > kijun):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Ichimoku_Cloud_TK_Cross"
timeframe = "6h"
leverage = 1.0