#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross + Volume Confirmation
Hypothesis: Ichimoku cloud provides dynamic support/resistance and trend direction. TK cross signals momentum shifts. Volume confirms institutional participation. Works in both bull/bear markets by adapting to cloud thickness and price/cloud relationship. Low trade frequency due to strict cloud/TK cross requirements.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Ichimoku components:
    - tenkan_sen: (9-period high + 9-period low)/2
    - kijun_sen: (26-period high + 26-period low)/2
    - senkou_span_a: (tenkan_sen + kijun_sen)/2 shifted 26 periods ahead
    - senkou_span_b: (52-period high + 52-period low)/2 shifted 26 periods ahead
    - chikou_span: close shifted 26 periods behind
    """
    def _rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def _rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    # Tenkan-sen (Conversion Line): 9-period high-low midpoint
    tenkan_sen = (_rolling_max(high, 9) + _rolling_min(low, 9)) / 2
    
    # Kijun-sen (Base Line): 26-period high-low midpoint
    kijun_sen = (_rolling_max(high, 26) + _rolling_min(low, 26)) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods forward
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): 52-period high-low midpoint shifted 26 periods forward
    senkou_span_b = (_rolling_max(high, 52) + _rolling_min(low, 52)) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods backward
    chikou_span = np.full_like(close, np.nan)
    for i in range(26, len(close)):
        chikou_span[i] = close[i - 26]
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_sen_1d, kijun_sen_1d, senkou_span_a_1d, senkou_span_b_1d, chikou_span_1d = calculate_ichimoku(
        high_1d, low_1d, close_1d
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for Ichimoku (need 52 + 26 periods)
    
    for i in range(start_idx, n):
        # Skip if any Ichimoku component is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(chikou_span_aligned[i])):
            signals[i] = 0.0
            continue
        
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        senkou_a = senkou_span_a_aligned[i]
        senkou_b = senkou_span_b_aligned[i]
        chikou = chikou_span_aligned[i]
        current_close = close[i]
        vol_ok = vol_spike[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        # TK Cross conditions
        tk_cross_up = tenkan > kijun and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan < kijun and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 0:
            # Enter long: price above cloud + TK cross up + chikou above price (26 periods ago) + volume spike
            if (current_close > cloud_top and 
                tk_cross_up and 
                chikou > close[i-26] if i >= 26 else False and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price below cloud + TK cross down + chikou below price (26 periods ago) + volume spike
            elif (current_close < cloud_bottom and 
                  tk_cross_down and 
                  chikou < close[i-26] if i >= 26 else False and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below cloud or TK cross down
            if current_close < cloud_bottom or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above cloud or TK cross up
            if current_close > cloud_top or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_Volume"
timeframe = "6h"
leverage = 1.0