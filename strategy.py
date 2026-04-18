#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTkCross
Hypothesis: Use 1d Ichimoku Tenkan/Kijun cross for direction, filter by 1d cloud (Senkou A/B) on 6b timeframe.
Only take long when price > cloud and TK cross bullish, short when price < cloud and TK cross bearish.
Ichimoku provides dynamic support/resistance and trend strength, working in both trending and ranging markets.
Targets 15-30 trades/year on 6h with strict entry conditions to avoid overtrading.
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
    
    # Get 1d data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan = np.full(len(close_1d), np.nan)
    for i in range(period_tenkan - 1, len(close_1d)):
        tenkan[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                     np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun = np.full(len(close_1d), np.nan)
    for i in range(period_kijun - 1, len(close_1d)):
        kijun[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                    np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + period_kijun  # 26 periods ahead
            if idx < len(senkou_a):
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    senkou_b = np.full(len(close_1d), np.nan)
    for i in range(period_senkou_b - 1, len(close_1d)):
        sb_value = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                    np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
        idx = i + period_kijun  # 26 periods ahead
        if idx < len(senkou_b):
            senkou_b[idx] = sb_value
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # TK cross signals: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish = tenkan_aligned > kijun_aligned
    tk_bearish = tenkan_aligned < kijun_aligned
    
    # Cloud: green when Senkou A > Senkou B, red when Senkou A < Senkou B
    # For filtering: price above cloud (bullish) when price > max(Senkou A, Senkou B)
    # Price below cloud (bearish) when price < min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Ichimoku calculations are valid
    start_idx = max(period_kijun + period_senkou_b, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above cloud + TK bullish cross
            if price_above_cloud[i] and tk_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK bearish cross
            elif price_below_cloud[i] and tk_bearish[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below cloud or TK turns bearish
            if (not price_above_cloud[i]) or (not tk_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above cloud or TK turns bullish
            if (not price_below_cloud[i]) or (not tk_bearish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTkCross"
timeframe = "6h"
leverage = 1.0