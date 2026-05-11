#!/usr/bin/env python3
name = "6h_1w_Ichimoku_Cloud"
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
    
    # Get weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Ichimoku components (weekly)
    high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    high_52 = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    
    tenkan = (high_9 + low_9) / 2
    kijun = (high_26 + low_26) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high_52 + low_52) / 2
    
    # Align to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 52)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: TK cross above cloud with volume surge
            if (tenkan_aligned[i] > kijun_aligned[i] and
                close[i] > cloud_top and
                volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: TK cross below cloud with volume surge
            elif (tenkan_aligned[i] < kijun_aligned[i] and
                  close[i] < cloud_bottom and
                  volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross below or price below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] or
                close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross above or price above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] or
                close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals