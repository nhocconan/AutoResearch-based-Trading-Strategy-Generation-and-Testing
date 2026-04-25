#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1
Hypothesis: Use Ichimoku Tenkan-Kijun cross on 6h with 1d cloud filter and volume confirmation. 
In bullish 1d trend (price above 1d cloud), buy when Tenkan crosses above Kijun on 6h; 
in bearish 1d trend (price below 1d cloud), sell when Tenkan crosses below Kijun on 6h. 
Volume spike (2.0x 20-bar avg) confirms institutional interest. 
Designed for 6h timeframe with moderate entries (~20-40/year) to balance signal quality and fee drag.
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
    
    # Get 1d data for HTF cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For simplicity, we use the midpoint as cloud reference
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    cloud_mid_1d = (cloud_top_1d + cloud_bottom_1d) / 2
    
    # Align 1d cloud midpoint to 6h timeframe
    cloud_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_mid_1d)
    
    # Calculate 6h Ichimoku components for TK cross
    # Tenkan-sen (6h): (9-period high + 9-period low)/2
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (6h): (26-period high + 26-period low)/2
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # TK cross signals
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for indicators
    start_idx = max(30, 26)  # volume MA(20) and Kijun(26)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cloud_mid_1d_aligned[i]) or 
            np.isnan(tenkan_6h[i]) or
            np.isnan(kijun_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d trend relative to cloud
        price_above_cloud = close[i] > cloud_mid_1d_aligned[i]
        price_below_cloud = close[i] < cloud_mid_1d_aligned[i]
        
        if position == 0:
            # Look for TK cross with volume confirmation and cloud filter
            long_signal = tk_cross_above[i] and volume_spike[i] and price_above_cloud
            short_signal = tk_cross_below[i] and volume_spike[i] and price_below_cloud
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when TK cross reverses or price goes below cloud
            exit_signal = tk_cross_below[i] or (close[i] < cloud_mid_1d_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when TK cross reverses or price goes above cloud
            exit_signal = tk_cross_above[i] or (close[i] > cloud_mid_1d_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_v1"
timeframe = "6h"
leverage = 1.0