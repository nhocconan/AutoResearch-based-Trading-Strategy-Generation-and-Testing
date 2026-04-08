#!/usr/bin/env python3
# [24947] 6h_1d_ichimoku_trend_follow_v1
# Hypothesis: 6-hour Ichimoku system with 1-day trend filter. Uses daily Kumo (cloud) as trend bias and
# Tenkan/Kijun cross on 6h for entry. In bullish cloud (price > cloud), look for TK cross up to go long.
# In bearish cloud (price < cloud), look for TK cross down to go short. Avoids chop by requiring cloud
# thickness > 0.1% of price. Works in both bull/bear by following higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1-day data for Ichimoku (cloud)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    tenkan_1d = np.full(len(close_1d), np.nan)
    for i in range(8, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    kijun_1d = np.full(len(close_1d), np.nan)
    for i in range(25, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = np.full(len(close_1d), np.nan)
    for i in range(25, len(close_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_a_1d[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    senkou_b_1d = np.full(len(close_1d), np.nan)
    for i in range(51, len(close_1d)):
        senkou_b_1d[i] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): not used for trend bias
    
    # Determine cloud (Kumo) and its color
    # Bullish cloud: Senkou A > Senkou B
    # Bearish cloud: Senkou A < Senkou B
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    cloud_bullish_1d = senkou_a_1d > senkou_b_1d  # True if bullish
    
    # Align Ichimoku components to 6-hour timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    cloud_bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish_1d.astype(float))
    
    # Calculate Tenkan and Kijun on 6h for entry signals
    tenkan_6h = np.full(n, np.nan)
    kijun_6h = np.full(n, np.nan)
    for i in range(8, n):
        tenkan_6h[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    for i in range(25, n):
        kijun_6h[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(26, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Cloud thickness filter: avoid chop when cloud is too thin
        cloud_thickness = abs(cloud_top_1d_aligned[i] - cloud_bottom_1d_aligned[i])
        price = close[i]
        if cloud_thickness < 0.001 * price:  # Less than 0.1% of price
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend bias from daily Ichimoku
        # Bullish bias: price above cloud AND cloud is bullish
        # Bearish bias: price below cloud AND cloud is bearish
        bullish_bias = (price > cloud_top_1d_aligned[i]) and (cloud_bullish_1d_aligned[i] > 0.5)
        bearish_bias = (price < cloud_bottom_1d_aligned[i]) and (cloud_bullish_1d_aligned[i] < 0.5)
        
        if position == 1:  # Long
            # Exit: TK cross down OR price re-enters cloud
            tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
            re_enter_cloud = (price >= cloud_bottom_1d_aligned[i]) and (price <= cloud_top_1d_aligned[i])
            if tk_cross_down or re_enter_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: TK cross up OR price re-enters cloud
            tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
            re_enter_cloud = (price >= cloud_bottom_1d_aligned[i]) and (price <= cloud_top_1d_aligned[i])
            if tk_cross_up or re_enter_cloud:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish bias AND TK cross up
            if bullish_bias:
                tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
                if tk_cross_up:
                    position = 1
                    signals[i] = 0.25
            # Enter short: bearish bias AND TK cross down
            elif bearish_bias:
                tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
                if tk_cross_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals