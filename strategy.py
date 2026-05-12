#!/usr/bin/env python3
name = "6h_Ichimoku_Cloud_Trend_Follow"
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
    
    # Load 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (10,26,52 periods)
    def ichimoku_cloud(high, low, close):
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
        period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
        tenkan = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
        period26_low = pd.Series(high).rolling(window=26, min_periods=26).min().values
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
        period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
        senkou_b = ((period52_high + period52_low) / 2)
        
        return tenkan, kijun, senkou_a, senkou_b
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_cloud(high_1d, low_1d, close_1d)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 0:
            # Long: Tenkan > Kijun (bullish cross) + price above cloud + volume filter
            if tenkan_1d_aligned[i] > kijun_1d_aligned[i] and close[i] > cloud_top and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun (bearish cross) + price below cloud + volume filter
            elif tenkan_1d_aligned[i] < kijun_1d_aligned[i] and close[i] < cloud_bottom and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan < Kijun (bearish cross) or price drops below cloud
            if tenkan_1d_aligned[i] < kijun_1d_aligned[i] or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan > Kijun (bullish cross) or price rises above cloud
            if tenkan_1d_aligned[i] > kijun_1d_aligned[i] or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals