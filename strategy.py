#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1w cloud filter and volume confirmation.
Uses weekly trend alignment (bullish/bearish cloud) to filter false crosses in ranging markets.
Volume spike (>2.0x 20-period median) confirms breakout strength.
Discrete sizing (0.25) minimizes fee drag. Target: 50-150 total trades over 4 years.
Works in bull/bear via 1w trend filter (cloud color) + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (2.0 * vol_median)
    
    # Load 1w data for HTF trend filter (Ichimoku cloud)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Align Ichimoku components to 6h timeframe (with proper look-ahead prevention)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a, additional_delay_bars=26)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b, additional_delay_bars=26)
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Bullish cloud: Senkou A > Senkou B
    bullish_cloud = senkou_a_aligned > senkou_b_aligned
    # Bearish cloud: Senkou A < Senkou B
    bearish_cloud = senkou_a_aligned < senkou_b_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_median[i]) or np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: TK cross bullish (Tenkan > Kijun) + price above cloud + bullish cloud + volume confirm
        if (tenkan_aligned[i] > kijun_aligned[i] and 
            close[i] > cloud_top[i] and 
            bullish_cloud[i] and 
            volume_confirm[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross bearish (Tenkan < Kijun) + price below cloud + bearish cloud + volume confirm
        elif (tenkan_aligned[i] < kijun_aligned[i] and 
              close[i] < cloud_bottom[i] and 
              bearish_cloud[i] and 
              volume_confirm[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits when price touches cloud bottom or TK cross turns bearish
        elif position == 1 and (close[i] <= cloud_bottom[i] or tenkan_aligned[i] < kijun_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Exit: short exits when price touches cloud top or TK cross turns bullish
        elif position == -1 and (close[i] >= cloud_top[i] or tenkan_aligned[i] > kijun_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1wTrend"
timeframe = "6h"
leverage = 1.0