# -*- coding: utf-8 -*-
# -*- mode: python; py-indent-offset: 2; -*-

#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend
Hypothesis: Ichimoku cloud from daily timeframe provides robust trend filter,
while 6h Tenkan-Kijun cross provides timely entry signals. Works in both bull
and bear markets by only taking trades in the direction of the daily cloud
(trend), reducing whipsaws. Uses volume confirmation to avoid false breaks.
Target: 20-50 trades/year (~80-200 over 4 years) with discrete sizing 0.25.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
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
    
    # Get 1d data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(9, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-9:i])
        period9_low[i] = np.min(low_1d[i-9:i])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(26, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-26:i])
        period26_low[i] = np.min(low_1d[i-26:i])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(52, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-52:i])
        period52_low[i] = np.min(low_1d[i-52:i])
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): not used for trend filter
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Determine cloud top and bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Trend filter: price above cloud = bullish, below cloud = bearish
    # Only take longs when price > cloud_top, shorts when price < cloud_bottom
    
    # 6h Tenkan/Kijun cross for entry signals
    # Tenkan-sen (9-period) on 6h
    period9_high_6h = np.full_like(high, np.nan)
    period9_low_6h = np.full_like(low, np.nan)
    for i in range(9, len(high)):
        period9_high_6h[i] = np.max(high[i-9:i])
        period9_low_6h[i] = np.min(low[i-9:i])
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (26-period) on 6h
    period26_high_6h = np.full_like(high, np.nan)
    period26_low_6h = np.full_like(low, np.nan)
    for i in range(26, len(high)):
        period26_high_6h[i] = np.max(high[i-26:i])
        period26_low_6h[i] = np.min(low[i-26:i])
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of Ichimoku periods)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge to avoid low-volume false signals
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above cloud, volume surge
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                close[i] > cloud_top[i] and volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun, price below cloud, volume surge
            elif (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                  close[i] < cloud_bottom[i] and volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Tenkan/Kijun cross in opposite direction OR price returns to cloud
            if position == 1:
                # Exit long: Tenkan crosses below Kijun OR price drops below cloud bottom
                if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]) or \
                   close[i] < cloud_bottom[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Tenkan crosses above Kijun OR price rises above cloud top
                if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]) or \
                   close[i] > cloud_top[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals