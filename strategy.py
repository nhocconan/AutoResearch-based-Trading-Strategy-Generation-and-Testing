#!/usr/bin/env python3
name = "6H_Ichimoku_Cloud_Filter_1dTrend_With_Volume"
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
    
    # Get daily data for Ichimoku cloud and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() +
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() +
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).max() +
                     pd.Series(low_1d).rolling(window=senkou_b_period, min_periods=senkou_b_period).min()) / 2
    
    # Current Kumo (cloud) top and bottom
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for indicators
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if Ichimoku data not ready
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average volume
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > avg_volume * 1.5
        else:
            volume_confirm = False
        
        if position == 0:
            # Enter long: TK cross up + price above cloud + volume confirmation
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and
                close[i] > kumo_top_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross down + price below cloud + volume confirmation
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and
                  close[i] < kumo_bottom_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross down or price below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]) or \
               close[i] < kumo_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up or price above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]) or \
               close[i] > kumo_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals