#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Signal_v1
Hypothesis: Use Ichimoku components from 1d timeframe for trend direction and 6s for entry timing.
Long when Tenkan > Kijun and price above Kumo cloud, short when opposite.
Kumo cloud acts as dynamic support/resistance. Tenkan/Kijun cross provides momentum signal.
Works in bull via trend continuation, in bear via counter-trend bounces off cloud.
Target: 50-150 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Signal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period).max() + 
                  pd.Series(low).rolling(window=tenkan_period).min()) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun_period).max() + 
                 pd.Series(low).rolling(window=kijun_period).min()) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou_span_b_period).max() + 
                      pd.Series(low).rolling(window=senkou_span_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-kijun_period)
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any data invalid
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine if price is above or below Kumo cloud
        top_kumo = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        bottom_kumo = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        price_above_kumo = close[i] > top_kumo
        price_below_kumo = close[i] < bottom_kumo
        
        # Tenkan/Kijun cross signals
        tenkan_above_kijun = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tenkan_below_kijun = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Entry conditions
        long_entry = price_above_kumo and tenkan_above_kijun
        short_entry = price_below_kumo and tenkan_below_kijun
        
        # Exit conditions: price crosses back into Kumo or Tenkan/Kijun reverse
        long_exit = price_below_kumo or tenkan_below_kijun
        short_exit = price_above_kumo or tenkan_above_kijun
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals