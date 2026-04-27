#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK cross filter from 1d for trend alignment
# Uses daily Ichimoku for trend direction and 6h Ichimoku for entry signals.
# Cloud filter avoids false signals in sideways markets, TK cross confirms momentum.
# Works in bull/bear by following daily trend while entering on 6h momentum shifts.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Get 6h data for Ichimoku entry signals
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan_1d = np.full(len(high_1d), np.nan)
    for i in range(period_tenkan - 1, len(high_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun_1d = np.full(len(high_1d), np.nan)
    for i in range(period_kijun - 1, len(high_1d)):
        kijun_1d[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_b_1d = np.full(len(high_1d), np.nan)
    for i in range(period_senkou_b - 1, len(high_1d)):
        senkou_b_1d[i] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Calculate Ichimoku components for 6h
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_6h = np.full(len(high_6h), np.nan)
    for i in range(period_tenkan - 1, len(high_6h)):
        tenkan_6h[i] = (np.max(high_6h[i-period_tenkan+1:i+1]) + np.min(low_6h[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_6h = np.full(len(high_6h), np.nan)
    for i in range(period_kijun - 1, len(high_6h)):
        kijun_6h[i] = (np.max(high_6h[i-period_kijun+1:i+1]) + np.min(low_6h[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_6h = (tenkan_6h + kijun_6h) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b_6h = np.full(len(high_6h), np.nan)
    for i in range(period_senkou_b - 1, len(high_6h)):
        senkou_b_6h[i] = (np.max(high_6h[i-period_senkou_b+1:i+1]) + np.min(low_6h[i-period_senkou_b+1:i+1])) / 2
    
    # Align all indicators to 6h timeframe
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    
    # Align 1d Ichimoku for trend filter
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(52, 52) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h_aligned[i]) or 
            np.isnan(kijun_6h_aligned[i]) or 
            np.isnan(senkou_a_6h_aligned[i]) or 
            np.isnan(senkou_b_6h_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or 
            np.isnan(senkou_b_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine 6h cloud color (green = bullish, red = bearish)
        # Cloud is between Senkou Span A and B
        cloud_top_6h = np.maximum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        cloud_bottom_6h = np.minimum(senkou_a_6h_aligned[i], senkou_b_6h_aligned[i])
        
        # Determine 1d cloud color for trend filter
        cloud_top_1d = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom_1d = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # 1d trend filter: price above/below cloud
        trend_up = price > cloud_top_1d
        trend_down = price < cloud_bottom_1d
        
        if position == 0:
            # Long: TK cross bullish + price above 6h cloud + 1d uptrend
            tk_cross_bullish = tenkan_6h_aligned[i] > kijun_6h_aligned[i]
            price_above_cloud = price > cloud_top_6h
            if tk_cross_bullish and price_above_cloud and trend_up:
                signals[i] = size
                position = 1
            # Short: TK cross bearish + price below 6h cloud + 1d downtrend
            elif tenkan_6h_aligned[i] < kijun_6h_aligned[i] and price < cloud_bottom_6h and trend_down:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TK cross bearish OR price below 6h cloud
            if tenkan_6h_aligned[i] < kijun_6h_aligned[i] or price < cloud_bottom_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: TK cross bullish OR price above 6h cloud
            if tenkan_6h_aligned[i] > kijun_6h_aligned[i] or price > cloud_top_6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dTrend"
timeframe = "6h"
leverage = 1.0