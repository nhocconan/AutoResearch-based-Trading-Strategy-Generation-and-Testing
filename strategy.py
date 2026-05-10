#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_20ATR
Hypothesis: Ichimoku TK cross with cloud filter from 1d timeframe, 
filtered by 20-period ATR volatility regime. Works in both bull/bear by 
using cloud as dynamic support/resistance and ATR to avoid choppy markets.
Target: 15-25 trades/year.
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
    
    # 1d Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    tenkan = np.full(len(high_1d), np.nan)
    for i in range(period_tenkan - 1, len(high_1d)):
        tenkan[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                     np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    kijun = np.full(len(high_1d), np.nan)
    for i in range(period_kijun - 1, len(high_1d)):
        kijun[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                    np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_a = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < len(high_1d):
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    period_senkou_b = 52
    senkou_b = np.full(len(high_1d), np.nan)
    for i in range(period_senkou_b - 1, len(high_1d)):
        senkou_b[i] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                       np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    # Shift Senkou B by 52 periods
    senkou_b_shifted = np.full(len(high_1d), np.nan)
    for i in range(len(senkou_b)):
        if not np.isnan(senkou_b[i]):
            idx = i + 52
            if idx < len(high_1d):
                senkou_b_shifted[idx] = senkou_b[i]
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    # ATR(20) for volatility regime filter
    atr_period = 20
    tr = np.maximum(high[1:] - low[1:], 
                    np.abs(high[1:] - close[:-1]), 
                    np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.mean(tr[i - atr_period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 52 + 26)  # Ensure Ichimoku and ATR are ready
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or \
           np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # ATR regime: only trade when volatility is elevated (avoid chop)
        atr_ma = np.full(n, np.nan)
        if i >= 50:
            atr_ma[i] = np.mean(atr[max(0, i-49):i+1])
            atr_ratio = atr[i] / atr_ma[i] if atr_ma[i] > 0 else 1.0
            volatility_filter = atr_ratio > 1.2  # Trade only when volatility is above average
        else:
            volatility_filter = False
        
        if position == 0:
            # Long: TK cross bullish AND price above cloud
            if tenkan_aligned[i] > kijun_aligned[i] and close[i] > cloud_top and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish AND price below cloud
            elif tenkan_aligned[i] < kijun_aligned[i] and close[i] < cloud_bottom and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish OR price drops below cloud bottom
            if tenkan_aligned[i] < kijun_aligned[i] or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish OR price rises above cloud top
            if tenkan_aligned[i] > kijun_aligned[i] or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals