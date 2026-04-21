#!/usr/bin/env python3
"""
6h_1d_IchimokuCloud_KijunBase
Hypothesis: Use Ichimoku cloud from 1d to filter trend direction (price above/below cloud) and 
Tenkan/Kijun cross for entry timing on 6h. Cloud acts as dynamic support/resistance.
In bull markets: price above cloud + bullish TK cross = long. 
In bear markets: price below cloud + bearish TK cross = short.
Kijun base acts as dynamic trailing stop. Designed for ~15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou"""
    n = len(high)
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    senkou_a = np.full(n, np.nan)
    senkou_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9 = 9
    for i in range(period9 - 1, n):
        high9 = np.max(high[i-period9+1:i+1])
        low9 = np.min(low[i-period9+1:i+1])
        tenkan[i] = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26 = 26
    for i in range(period26 - 1, n):
        high26 = np.max(high[i-period26+1:i+1])
        low26 = np.min(low[i-period26+1:i+1])
        kijun[i] = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52 = 52
    for i in range(period52 - 1, n):
        high52 = np.max(high[i-period52+1:i+1])
        low52 = np.min(low[i-period52+1:i+1])
        senkou_b[i] = (high52 + low52) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Cloud top/bottom (Senkou A/B)
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align to 6h timeframe (wait for 1d close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top_1d_aligned[i]
        price_below_cloud = price < cloud_bottom_1d_aligned[i]
        
        # Tenkan/Kijun cross
        tk_cross_bullish = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_bearish = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        if position == 0:
            # Long: price above cloud + bullish TK cross
            if price_above_cloud and tk_cross_bullish:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + bearish TK cross
            elif price_below_cloud and tk_cross_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below cloud OR bearish TK cross
            if price_below_cloud or tk_cross_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above cloud OR bullish TK cross
            if price_above_cloud or tk_cross_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_IchimokuCloud_KijunBase"
timeframe = "6h"
leverage = 1.0