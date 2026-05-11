#!/usr/bin/env python3
name = "6h_12h_Ichimoku_TK_Cross_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (standard periods)
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2, plotted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + low)/2, plotted 26 periods ahead
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    
    # Calculate Tenkan-sen
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Calculate Kijun-sen
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Calculate Senkou Span A
    senkou_a = (tenkan + kijun) / 2
    
    # Calculate Senkou Span B
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    # Tenkan and Kijun are current values (no forward shift needed for signal)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    # Senkou Span A and B are plotted 26 periods ahead, so we need to shift back
    # to get their current values for cloud calculation
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom (Senkou Span A and B form the cloud)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # TK Cross signals
    tk_cross_above = tenkan_aligned > kijun_aligned
    tk_cross_below = tenkan_aligned < kijun_aligned
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume moving average (20-period) for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK cross bullish + price above cloud + volume confirmation
            if (tk_cross_above[i] and 
                price_above_cloud[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish + price below cloud + volume confirmation
            elif (tk_cross_below[i] and 
                  price_below_cloud[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross bearish or price drops below cloud
            if (tk_cross_below[i] or not price_above_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish or price rises above cloud
            if (tk_cross_above[i] or not price_below_cloud[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals