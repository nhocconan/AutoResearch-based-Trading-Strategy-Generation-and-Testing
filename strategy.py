#!/usr/bin/env python3
# 6H_Ichimoku_Cloud_Twist_With_Volume
# Hypothesis: Combines Ichimoku cloud twist (Tenkan/Kijun cross) with cloud color from 1d timeframe
# and volume confirmation to capture trend changes with low frequency.
# Cloud twist signals momentum shift; cloud color from higher timeframe filters counter-trend trades.
# Works in bull markets (green cloud + bullish twist) and bear markets (red cloud + bearish twist).
# Targets 15-35 trades/year to avoid fee drag.

name = "6H_Ichimoku_Cloud_Twist_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52)
    tenkan = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
              pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
             pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low).rolling(window=52, min_periods=52).min()) / 2)
    
    # Shift Senkou spans forward by 26 periods (cloud ahead)
    senkou_a = np.roll(senkou_a, 26)
    senkou_b = np.roll(senkou_b, 26)
    senkou_a[:26] = np.nan
    senkou_b[:26] = np.nan
    
    # Cloud color: green if Senkou A > Senkou B, red otherwise
    cloud_green = senkou_a > senkou_b
    
    # Twist: Tenkan/Kijun cross
    twist_up = tenkan > kijun
    twist_down = tenkan < kijun
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Daily trend filter: cloud color from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    senkou_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                    pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    senkou_a_1d = np.roll(senkou_a_1d, 26)
    senkou_b_1d = np.roll(senkou_b_1d, 26)
    senkou_a_1d[:26] = np.nan
    senkou_b_1d[:26] = np.nan
    
    cloud_green_1d = senkou_a_1d > senkou_b_1d
    cloud_green_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_green_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough history for Ichimoku
    
    for i in range(start_idx, n):
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(vol_threshold[i]) or np.isnan(cloud_green_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_green_cloud = cloud_green_1d_aligned[i] > 0.5  # True if 1d cloud is green
        is_red_cloud = cloud_green_1d_aligned[i] <= 0.5   # True if 1d cloud is red
        
        if position == 0:
            # Long entry: Green cloud from 1d + bullish twist + volume confirmation
            if is_green_cloud and twist_up[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Red cloud from 1d + bearish twist + volume confirmation
            elif is_red_cloud and twist_down[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Red cloud or bearish twist
            if is_red_cloud or twist_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Green cloud or bullish twist
            if is_green_cloud or twist_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals