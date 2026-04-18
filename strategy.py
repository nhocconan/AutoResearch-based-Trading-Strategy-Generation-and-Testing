#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d Trend Filter + Volume Confirmation
Hypothesis: Ichimoku provides a comprehensive trend system (Tenkan/Kijun cross, cloud support/resistance).
Combined with 1d trend filter (price above/below 1d Kumo) and volume confirmation, it captures strong
trends while avoiding false signals in ranging markets. Designed for 6h timeframe to balance signal
quality and trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over tenkan period
    tenkan_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max()
    tenkan_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over kijun period
    kijun_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max()
    kijun_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min()
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted kijun periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over senkou period shifted kijun ahead
    senkou_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max()
    senkou_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min()
    senkou_b = ((senkou_high + senkou_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted back kijun periods
    chikou = close  # We'll handle shifting in alignment
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        high_1d, low_1d, close_1d, tenkan=9, kijun=26, senkou=52
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate 1d trend filter: price above/below cloud
    # Cloud top is max(senkou_a, senkou_b), cloud bottom is min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Tenkan/Kijun cross signals
    tenkan_above_kijun = tenkan_1d_aligned > kijun_1d_aligned
    tenkan_below_kijun = tenkan_1d_aligned < kijun_1d_aligned
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: Tenkan crosses above Kijun AND price above cloud AND volume spike
            if (tenkan_above_kijun[i] and 
                price_above_cloud[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan crosses below Kijun AND price below cloud AND volume spike
            elif (tenkan_below_kijun[i] and 
                  price_below_cloud[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun OR price falls below cloud
            if (tenkan_below_kijun[i] or 
                close[i] < cloud_bottom[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun OR price rises above cloud
            if (tenkan_above_kijun[i] or 
                close[i] > cloud_top[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0