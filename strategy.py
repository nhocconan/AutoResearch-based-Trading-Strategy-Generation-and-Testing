#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_Volume
Hypothesis: Use Ichimoku TK cross (Tenkan/Kijun) on 6h with 1d cloud filter and volume confirmation.
Trades in direction of 1d trend (price above/below cloud) with momentum confirmation from TK cross.
Works in bull via breakouts above cloud, bear via breakdowns below cloud.
Target: 15-30 trades/year to minimize fee drag.
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    highest_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan = (highest_9 + lowest_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    highest_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun = (highest_26 + lowest_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    highest_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_52 + lowest_52) / 2
    
    # 1d trend filter: price above/below cloud from daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < senkou_span_b_period:
        return np.zeros(n)
    
    # Calculate 1d Ichimoku cloud
    highest_9_1d = pd.Series(df_1d['high']).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_9_1d = pd.Series(df_1d['low']).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_1d = (highest_9_1d + lowest_9_1d) / 2
    
    highest_26_1d = pd.Series(df_1d['high']).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_26_1d = pd.Series(df_1d['low']).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_1d = (highest_26_1d + lowest_26_1d) / 2
    
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    
    highest_52_1d = pd.Series(df_1d['high']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_52_1d = pd.Series(df_1d['low']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1d = (highest_52_1d + lowest_52_1d) / 2
    
    # Cloud top and bottom (Senkou Span A and B)
    cloud_top_1d = np.maximum(senkou_span_a_1d, senkou_span_b_1d)
    cloud_bottom_1d = np.minimum(senkou_span_a_1d, senkou_span_b_1d)
    
    # Align 1d cloud to 6s timeframe
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Ichimoku calculations
    start_idx = max(senkou_span_b_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # TK cross signals
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price position relative to 1d cloud
        price_above_cloud = close[i] > cloud_top_aligned[i]
        price_below_cloud = close[i] < cloud_bottom_aligned[i]
        
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: TK cross up + price above 1d cloud + volume confirmation
            if tk_cross_up and price_above_cloud and vol_conf:
                signals[i] = size
                position = 1
            # Short: TK cross down + price below 1d cloud + volume confirmation
            elif tk_cross_down and price_below_cloud and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK cross down or price falls below cloud bottom
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK cross up or price rises above cloud top
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0