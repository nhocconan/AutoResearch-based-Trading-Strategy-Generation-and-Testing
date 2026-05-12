#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (standard settings: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high9 + low9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high26 + low26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high52 + low52) / 2)
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Volume spike on 6h: current volume > 1.8x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, volume spike
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                close[i] > cloud_top and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, volume spike
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  close[i] < cloud_bottom and 
                  vol_spike[i]):
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