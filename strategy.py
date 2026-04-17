#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line)
    high_tenkan = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line)
    high_kijun = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values
    low_kijun = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B)
    high_senkou_b = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, with volume
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > cloud_top and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, with volume
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < cloud_bottom and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Tenkan crosses below Kijun or price falls below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or 
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Tenkan crosses above Kijun or price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or 
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_Volume"
timeframe = "6h"
leverage = 1.0