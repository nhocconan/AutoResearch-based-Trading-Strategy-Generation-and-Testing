#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend
Hypothesis: 6h Ichimoku Tenkan-Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation.
Ichimoku provides dynamic support/resistance and trend direction. The cloud acts as a strong trend filter - 
price above cloud = bullish bias, below cloud = bearish bias. TK cross within cloud-filtered direction 
captures momentum shifts with reduced whipsaw. Volume confirmation ensures institutional participation.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
Works in both bull and bear markets by aligning with higher timeframe trend filter.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (max of Ichimoku periods)
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Determine cloud boundaries (Senkou Span A/B form the cloud)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Bullish signal: TK cross up + price above cloud + volume spike
        tenkan_prev = tenkan_sen_aligned[i-1]
        kijun_prev = kijun_sen_aligned[i-1]
        tenkan_curr = tenkan_sen_aligned[i]
        kijun_curr = kijun_sen_aligned[i]
        
        tk_cross_up = (tenkan_prev <= kijun_prev) and (tenkan_curr > kijun_curr)
        price_above_cloud = close[i] > cloud_top
        
        # Bearish signal: TK cross down + price below cloud + volume spike
        tk_cross_down = (tenkan_prev >= kijun_prev) and (tenkan_curr < kijun_curr)
        price_below_cloud = close[i] < cloud_bottom
        
        # Long logic
        if tk_cross_up and price_above_cloud and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic
        elif tk_cross_down and price_below_cloud and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross in opposite direction OR price crosses cloud
        elif position == 1 and (tk_cross_down or close[i] < cloud_bottom):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tk_cross_up or close[i] > cloud_top):
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

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0