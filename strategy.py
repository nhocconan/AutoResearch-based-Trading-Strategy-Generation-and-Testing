#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_v1
Ichimoku-based strategy using daily Ichimoku components:
- Tenkan-sen (9) and Kijun-sen (26) cross for entry signal
- Senkou Span A/B from 26 periods ahead form the cloud
- Price must break above/below cloud with Tenkan/Kijun cross
- Volume confirmation: current volume > 1.5x 20-period average
- Exit when price re-enters cloud or Tenkan/Kijun cross reverses
Designed to capture trend breaks while avoiding false signals in chop.
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
    volume = prices['volume'].values
    
    # === Daily Ichimoku components ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 plotted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align all Ichimoku components to 6h timeframe (wait for 1d close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    
    # Volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period: need enough data for Ichimoku calculations
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above cloud AND Tenkan crosses above Kijun
            if (close[i] > upper_cloud and 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i] and
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and  # cross just happened
                vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below cloud AND Tenkan crosses below Kijun
            elif (close[i] < lower_cloud and 
                  tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and  # cross just happened
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price re-enters cloud or Tenkan/Kijun cross reverses
        elif position == 1:
            # Exit long: price re-enters cloud OR Tenkan crosses below Kijun
            if (close[i] <= upper_cloud or 
                tenkan_sen_aligned[i] < kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters cloud OR Tenkan crosses above Kijun
            if (close[i] >= lower_cloud or 
                tenkan_sen_aligned[i] > kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0