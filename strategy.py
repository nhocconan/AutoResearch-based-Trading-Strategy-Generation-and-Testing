#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 1d cloud filter (price above/below cloud) and volume confirmation (1.5x average). 
This strategy captures momentum in both bull and bear markets by aligning with the higher timeframe trend (cloud) while using the TK cross for precise entry. 
Volume confirmation reduces false signals. Targets 50-150 trades over 4 years with discrete position sizing (0.25).
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
    
    # Load 1d data ONCE before loop for Ichimoku components and cloud
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Cloud: price above Senkou Span A AND Senkou Span B = bullish
    # price below Senkou Span A AND Senkou Span B = bearish
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan_sen_aligned > kijun_sen_aligned) & (tenkan_sen_aligned <= kijun_sen_aligned)
    tk_cross_down = (tenkan_sen_aligned < kijun_sen_aligned) & (tenkan_sen_aligned >= kijun_sen_aligned)
    
    # Volume filter: volume > 1.5 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou Span B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: TK cross up AND price above cloud AND volume confirmation
            if tk_cross_up[i] and price_above_cloud[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross down AND price below cloud AND volume confirmation
            elif tk_cross_down[i] and price_below_cloud[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross down OR price falls below cloud
            if tk_cross_down[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross up OR price rises above cloud
            if tk_cross_up[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0