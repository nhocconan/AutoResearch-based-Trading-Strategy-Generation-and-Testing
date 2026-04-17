#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Ichimoku cloud from 1d timeframe (Tenkan/Kijun/Senkou Span A/B) + TK cross on 6h.
Long when price above cloud + TK cross bullish, short when price below cloud + TK cross bearish.
Exit when price enters cloud or TK cross reverses.
Designed to capture major trends with cloud acting as dynamic support/resistance.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === TK Cross on 6h (Tenkan-sen/Kijun-sen) ===
    # Tenkan-sen: (9-period high + low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen: (26-period high + low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    tk_cross = tenkan - kijun  # >0 bullish, <0 bearish
    
    # === Ichimoku Cloud from 1d timeframe ===
    df_1d = get_htf_data(prices, '1d')
    
    # Tenkan-sen (conversion line) - 9-period
    high_9_1d = pd.Series(df_1d['high'].values).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(df_1d['low'].values).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2.0
    
    # Kijun-sen (base line) - 26-period
    high_26_1d = pd.Series(df_1d['high'].values).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(df_1d['low'].values).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2.0
    
    # Senkou Span A (leading span A) - (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2.0)
    
    # Senkou Span B (leading span B) - (52-period high + low)/2 shifted 26 periods ahead
    high_52_1d = pd.Series(df_1d['high'].values).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(df_1d['low'].values).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2.0
    
    # Align all Ichimoku components to 6h timeframe (wait for daily close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60  # enough for Ichimoku calculations
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tk_cross[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above cloud + TK cross bullish
            if (close[i] > cloud_top[i] and 
                tk_cross[i] > 0):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below cloud + TK cross bearish
            elif (close[i] < cloud_bottom[i] and 
                  tk_cross[i] < 0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price enters cloud OR TK cross turns bearish
            if (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]) or \
               tk_cross[i] < 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price enters cloud OR TK cross turns bullish
            if (close[i] <= cloud_top[i] and close[i] >= cloud_bottom[i]) or \
               tk_cross[i] > 0:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0