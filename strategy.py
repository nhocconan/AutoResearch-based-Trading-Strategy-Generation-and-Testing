#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1d
Hypothesis: Ichimoku Tenkan-Kijun cross with daily cloud filter on 6h timeframe.
Long when TK crosses above AND price above cloud (Senkou Span A/B). Short when TK crosses below AND price below cloud.
Uses 1d Ichimoku for cloud to avoid repainting and ensure stability. Targets 12-30 trades/year.
Works in trending markets (bull/bear) by following cloud direction; avoids ranges via TK cross confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get daily data for Ichimoku cloud (more stable)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate Ichimoku on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 6h TK cross
    tenkan_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    tenkan_6h = (tenkan_6h + pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    kijun_6h = (kijun_6h + pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    tk_cross_above = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))
    
    # Cloud top and bottom (Senkou A/B)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tk_cross_above[i]) if i < len(tk_cross_above) else True or
            np.isnan(tk_cross_below[i]) if i < len(tk_cross_below) else True):
            signals[i] = 0.0
            continue
        
        # Price above/below cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # TK cross signals
        long_signal = tk_cross_above[i] and price_above_cloud
        short_signal = tk_cross_below[i] and price_below_cloud
        
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif not price_above_cloud and not price_below_cloud and position != 0:
            # Price in cloud - exit
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1d"
timeframe = "6h"
leverage = 1.0