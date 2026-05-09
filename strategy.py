#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (standard parameters: 9, 26, 52)
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    
    # Align to 6h
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 52 + 26  # Need enough data for Senkou B calculation
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        tenkan = tenkan_6h[i]
        kijun = kijun_6h[i]
        senkou_a = senkou_a_6h[i]
        senkou_b = senkou_b_6h[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)
        
        if position == 0:
            # Enter long: TK cross above AND price above cloud
            if tenkan > kijun and close[i] > cloud_top:
                signals[i] = 0.25
                position = 1
            # Enter short: TK cross below AND price below cloud
            elif tenkan < kijun and close[i] < cloud_bottom:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TK cross below OR price drops below cloud
            if tenkan < kijun or close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross above OR price rises above cloud
            if tenkan > kijun or close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals