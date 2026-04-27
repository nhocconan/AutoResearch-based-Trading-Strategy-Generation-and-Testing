#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) on 6h with 1d cloud filter (price above/below Kumo) captures medium-term momentum while avoiding counter-trend whipsaws. Volume confirmation adds robustness. Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for cloud (Kumo) filter
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components on 6h (primary timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # 1d Cloud (Kumo) filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan and Kijun for cloud calculation
    high_1d_tenkan = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_1d_tenkan = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_1d_tenkan + low_1d_tenkan) / 2
    
    high_1d_kijun = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_1d_kijun = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_1d_kijun + low_1d_kijun) / 2
    
    # 1d Senkou Span A and B (shifted forward 26 periods)
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    high_1d_senkou_b = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_1d_senkou_b = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_1d_senkou_b + low_1d_senkou_b) / 2
    
    # Align 1d cloud components to 6h timeframe
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 52-period Senkou B, 26-period Kijun, 20-period volume avg
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        senkou_a_1d_val = senkou_a_1d_aligned[i]
        senkou_b_1d_val = senkou_b_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Determine cloud boundaries (Senkou A and B)
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # Determine 1d cloud boundaries
        upper_cloud_1d = max(senkou_a_1d_val, senkou_b_1d_val)
        lower_cloud_1d = min(senkou_a_1d_val, senkou_b_1d_val)
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long conditions: TK cross up, price above both clouds, volume confirmation
            if (tk_cross_up and 
                close_val > upper_cloud and 
                close_val > upper_cloud_1d and 
                vol_conf):
                signals[i] = size
                position = 1
            # Short conditions: TK cross down, price below both clouds, volume confirmation
            elif (tk_cross_down and 
                  close_val < lower_cloud and 
                  close_val < lower_cloud_1d and 
                  vol_conf):
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK cross down or price drops below cloud
            if tk_cross_down or close_val < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross up or price rises above cloud
            if tk_cross_up or close_val > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloudFilter"
timeframe = "6h"
leverage = 1.0