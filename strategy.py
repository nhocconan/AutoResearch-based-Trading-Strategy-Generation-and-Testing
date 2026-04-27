#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: Uses Ichimoku cloud from 1d timeframe as trend filter and support/resistance, with 6h price breaking above/below cloud edges for entries. Volume confirmation filters false breakouts. Works in bull/bear by following 1d trend while using cloud for dynamic S/R.
Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.
"""

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
    
    # Get 1d data for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku parameters: tenkan=9, kijun=26, senkou_span_b=52, displacement=26
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe (displaced by 26 periods)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen, additional_delay_bars=26)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen, additional_delay_bars=26)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b, additional_delay_bars=26)
    
    # Cloud edges: upper cloud = max(Senkou A, Senkou B), lower cloud = min(Senkou A, Senkou B)
    upper_cloud = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_cloud = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Ichimoku (52) + displacement (26) + volume avg (20)
    start_idx = max(52 + 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        upper_cloud_val = upper_cloud[i]
        lower_cloud_val = lower_cloud[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: price breaks above/below cloud with TK cross AND volume
            # Long: price > upper cloud AND Tenkan > Kijun (bullish TK cross) AND volume
            long_condition = (close_val > upper_cloud_val) and (tenkan_val > kijun_val) and vol_conf
            # Short: price < lower cloud AND Tenkan < Kijun (bearish TK cross) AND volume
            short_condition = (close_val < lower_cloud_val) and (tenkan_val < kijun_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to cloud OR TK cross turns bearish
            exit_condition = (close_val <= upper_cloud_val) or (tenkan_val < kijun_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to cloud OR TK cross turns bullish
            exit_condition = (close_val >= lower_cloud_val) or (tenkan_val > kijun_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0