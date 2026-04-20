#!/usr/bin/env python3
"""
4h_4h4h_IchimokuCloud_Breakout_With_Volume_Filter
Hypothesis: Trade 4h price breakouts above/below Ichimoku Cloud with volume confirmation and ATR-based stop. 
Long when price breaks above Kumo (cloud top) with volume spike; short when breaks below Kumo (cloud bottom) with volume spike.
Ichimoku Cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) acts as dynamic support/resistance. 
Volume filter reduces false breakouts. Works in bull/bear: cloud adapts to volatility, volume confirms breakout strength.
Target: 50-100 total trades over 4 years (12-25/year) with position size 0.25.
"""

name = "4h_4h4h_IchimokuCloud_Breakout_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    high_9 = rolling_max(high, tenkan_period)
    low_9 = rolling_min(low, tenkan_period)
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = rolling_max(high, kijun_period)
    low_26 = rolling_min(low, kijun_period)
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = rolling_max(high, senkou_period)
    low_52 = rolling_min(low, senkou_period)
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # The Kumo (Cloud) is between Senkou Span A and Senkou Span B
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = senkou_period  # Ensure Ichimoku components are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above cloud top with volume filter
            if close[i] > cloud_top[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud bottom with volume filter
            elif close[i] < cloud_bottom[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below cloud bottom
            if close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above cloud top
            if close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals