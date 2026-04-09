#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan = np.full(len(df_1d), np.nan)
    for i in range(period_tenkan - 1, len(df_1d)):
        tenkan[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                     np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun = np.full(len(df_1d), np.nan)
    for i in range(period_kijun - 1, len(df_1d)):
        kijun[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                    np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + period_kijun  # 26 periods ahead
            if idx < len(df_1d):
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    senkou_b = np.full(len(df_1d), np.nan)
    for i in range(period_senkou_b - 1, len(df_1d)):
        senkou_b[i] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                       np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
        idx = i + period_kijun  # 26 periods ahead
        if idx < len(df_1d):
            senkou_b[idx] = senkou_b[i]  # Assign the calculated value to the future position
    
    # Chikou Span (Lagging Span): Current close plotted 26 periods back
    chikou = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d) - period_kijun):
        chikou[i] = close_1d[i + period_kijun]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_6h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Volume confirmation: 4-period average (24h)
    vol_ma_4 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 4:
            vol_sum -= volume[i-4]
        if i >= 3:
            vol_ma_4[i] = vol_sum / 4
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup for Senkou B
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or 
            np.isnan(chikou_6h[i]) or np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        if position == 1:  # Long position
            # Exit: price closes below cloud OR Tenkan-Kijun cross down
            if (close[i] < cloud_bottom) or (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above cloud OR Tenkan-Kijun cross up
            if (close[i] > cloud_top) or (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above cloud, Tenkan > Kijun, Chikou above price from 26 periods ago
            vol_ratio = volume[i] / vol_ma_4[i] if vol_ma_4[i] > 0 else 0
            if (close[i] > cloud_top and 
                tenkan_6h[i] > kijun_6h[i] and 
                not np.isnan(chikou_6h[i]) and chikou_6h[i] > close[i] and
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price below cloud, Tenkan < Kijun, Chikou below price from 26 periods ago
            elif (close[i] < cloud_bottom and 
                  tenkan_6h[i] < kijun_6h[i] and 
                  not np.isnan(chikou_6h[i]) and chikou_6h[i] < close[i] and
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals