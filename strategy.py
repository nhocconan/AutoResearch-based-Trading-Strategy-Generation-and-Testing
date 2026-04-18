#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: Use 1d Ichimoku for trend direction and cloud filtering, with 6h Tenkan-Kijun cross for entry timing.
- Long when price > 1d Ichimoku cloud AND Tenkan crosses above Kijun on 6h
- Short when price < 1d Ichimoku cloud AND Tenkan crosses below Kijun on 6h
- Exit when price crosses opposite Tenkan/Kijun line or re-enters cloud
Ichimoku works well in trending markets (cloud acts as dynamic support/resistance) and ranges (TK cross signals reversals).
The 1d cloud filters for higher-timeframe trend, reducing false signals on 6h.
Target: 20-40 trades/year by requiring both TK cross and cloud alignment.
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
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9 = 9
    if n >= period9:
        for i in range(period9, n):
            tenkan[i] = (np.max(high[i-period9:i]) + np.min(low[i-period9:i])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26 = 26
    if n >= period26:
        for i in range(period26, n):
            kijun[i] = (np.max(high[i-period26:i]) + np.min(low[i-period26:i])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    if n >= period26:
        for i in range(n):
            if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
                idx = i + 26
                if idx < n:
                    senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52 = 52
    if n >= period52:
        for i in range(n):
            if i + period52 < n:
                senkou_b[i + 26] = (np.max(high[i:i+period52]) + np.min(low[i:i+period52])) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku (higher timeframe trend filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku on 1d
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h Tenkan and Kijun for entry signals
    period9 = 9
    period26 = 26
    tenkan_6h = np.full(n, np.nan)
    kijun_6h = np.full(n, np.nan)
    
    if n >= period9:
        for i in range(period9, n):
            tenkan_6h[i] = (np.max(high[i-period9:i]) + np.min(low[i-period9:i])) / 2
    
    if n >= period26:
        for i in range(period26, n):
            kijun_6h[i] = (np.max(high[i-period26:i]) + np.min(low[i-period26:i])) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period26, 26) + 1  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou A/B)
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Price relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        
        # 6h TK cross signals
        tk_cross_up = tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]
        tk_cross_down = tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]
        
        if position == 0:
            # Long: price above cloud AND TK cross up
            if price_above_cloud and tk_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND TK cross down
            elif price_below_cloud and tk_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below cloud OR TK cross down
            if price_below_cloud or tk_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above cloud OR TK cross up
            if price_above_cloud or tk_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0