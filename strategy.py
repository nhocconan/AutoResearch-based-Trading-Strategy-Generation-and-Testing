#!/usr/bin/env python3
"""
6h Ichimoku TK Cross + 1d Cloud Filter + Volume Spike
Hypothesis: Ichimoku Tenkan/Kijun cross on 6h provides momentum signals, while 
1d Ichimoku cloud acts as a trend filter (price above cloud = bull bias, below = bear bias).
Volume spike confirms breakout strength. Works in bull/bear via cloud position.
Target: 12-37 trades/year with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate 6h Ichimoku components (Tenkan: 9-period, Kijun: 26-period)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Calculate 1d Ichimoku cloud components
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    high_9_1d = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    senkou_a = ((tenkan_1d + kijun_1d) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52_1d = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52_1d + low_52_1d) / 2
    
    # Shift Senkou spans 26 periods ahead (for cloud)
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to shift
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Align 1d cloud to 6h timeframe
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_shifted)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for data to propagate
    start_idx = 60  # 26 (shift) + 26 (Kijun) + buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK Cross: Tenkan crossing above/below Kijun
        tk_cross_up = (tenkan_val > kijun_val) and (tenkan[i-1] <= kijun[i-1])
        tk_cross_down = (tenkan_val < kijun_val) and (tenkan[i-1] >= kijun[i-1])
        
        # Price relative to cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: TK cross up AND price above cloud AND volume spike
            if tk_cross_up and price_above_cloud and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND price below cloud AND volume spike
            elif tk_cross_down and price_below_cloud and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross down OR price drops below cloud
            if tk_cross_down or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross up OR price rises above cloud
            if tk_cross_up or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TKCross_CloudFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0