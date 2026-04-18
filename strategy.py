#!/usr/bin/env python3
"""
6h_MultiTF_Ichimoku_Cloud_Filter_V1
Hypothesis: Use daily Ichimoku cloud for directional bias (price above/below cloud) and Tenkan-Kijun cross for entry timing on 6h. The cloud acts as dynamic support/resistance, providing trend context that works in both bull and bear markets. TK cross gives timely entries within the trend. Add volume confirmation to avoid false breaks. Target 20-50 trades/year.
"""

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
    
    # Get daily data for Ichimoku (cloud and TK cross)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a, additional_delay_bars=26)  # shifted 26 days ahead
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b, additional_delay_bars=26)  # shifted 26 days ahead
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK cross signals
    tk_cross_above = tenkan_6h > kijun_6h
    tk_cross_below = tenkan_6h < kijun_6h
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above cloud + TK cross bullish + volume
            if close[i] > cloud_top[i] and tk_cross_above[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud + TK cross bearish + volume
            elif close[i] < cloud_bottom[i] and tk_cross_below[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below cloud or TK cross turns bearish
            if close[i] < cloud_top[i] or tk_cross_below[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above cloud or TK cross turns bullish
            if close[i] > cloud_bottom[i] or tk_cross_above[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MultiTF_Ichimoku_Cloud_Filter_V1"
timeframe = "6h"
leverage = 1.0