#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_20ATR
Hypothesis: Use Ichimoku cloud from 1d as trend filter, Tenkan-Kijun cross on 6h for entry, with ATR volatility filter.
Works in bull by following cloud direction, works in bear by shorting below cloud. Target: 12-30 trades/year.
"""

name = "6h_Ichimoku_Cloud_Filter_1dTrend_20ATR"
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
    
    # 1d Ichimoku components (Tenkan, Kijun, Senkou A/B)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan = np.full(len(high_1d), np.nan)
    for i in range(period_tenkan - 1, len(high_1d)):
        tenkan[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun = np.full(len(high_1d), np.nan)
    for i in range(period_kijun - 1, len(high_1d)):
        kijun[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < len(senkou_a):
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_b = np.full(len(high_1d), np.nan)
    for i in range(period_senkou_b - 1, len(high_1d)):
        val = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
        idx = i + 26
        if idx < len(senkou_b):
            senkou_b[idx] = val
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 6h ATR(20) for volatility filter
    atr_period = 20
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i - atr_period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52 + 26, atr_period)  # Senkou B needs 52+26 bars
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # Ichimoku signal: Tenkan-Kijun cross
        tk_cross = tenkan_6h[i] - kijun_6h[i]
        tk_cross_prev = tenkan_6h[i-1] - kijun_6h[i-1] if i > 0 else 0
        
        # Volatility filter: only trade when ATR > 0.5 * ATR mean (avoid low volatility chop)
        atr_mean = np.nanmean(atr[max(0, i-50):i+1])
        vol_filter = atr[i] > 0.5 * atr_mean if not np.isnan(atr_mean) else True
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud
            if tk_cross > 0 and tk_cross_prev <= 0 and close[i] > cloud_top and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud
            elif tk_cross < 0 and tk_cross_prev >= 0 and close[i] < cloud_bottom and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Tenkan crosses below Kijun OR price drops below cloud bottom
            if tk_cross < 0 and tk_cross_prev >= 0 or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Tenkan crosses above Kijun OR price rises above cloud top
            if tk_cross > 0 and tk_cross_prev <= 0 or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals