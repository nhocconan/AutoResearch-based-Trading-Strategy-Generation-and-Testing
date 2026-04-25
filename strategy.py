#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_v1
Hypothesis: Trade Ichimoku TK (Tenkan/Kijun) cross on 6h with 1d cloud filter (price above/below cloud) and volume confirmation. In bullish 1d cloud (price > Senkou Span A/B), buy TK cross up; in bearish 1d cloud (price < Senkou Span A/B), sell TK cross down. Uses volume spike (1.8x 24-bar avg) to confirm momentum. Designed for 6h timeframe with moderate entries (~80/year) to capture medium-term trends while avoiding chop via cloud filter. Works in bull/bear via cloud-defined regime.
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
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    chikou_period = 22
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max().values +
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max().values +
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values +
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values) / 2
    
    # Align Ichimoku components to 6h timeframe (cloud is forward-shifted, so we need to align properly)
    # For cloud, we use the values as-of the current 6h bar (no additional shift needed beyond Ichimoku's built-in shift)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: 1.8x 24-bar average volume (4 days on 6h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = senkou_span_b_period  # 52 bars for Senkou Span B
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d Ichimoku cloud and price position
        top_cloud = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        bottom_cloud = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_cloud = close[i] > top_cloud
        price_below_cloud = close[i] < bottom_cloud
        
        # TK Cross signals
        tk_cross_up = (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) and \
                      (tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1])
        tk_cross_down = (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) and \
                        (tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1])
        
        if position == 0:
            # Look for TK cross with volume confirmation and cloud filter
            long_signal = tk_cross_up and price_above_cloud and volume_spike[i]
            short_signal = tk_cross_down and price_below_cloud and volume_spike[i]
            
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
            # Exit when price falls below cloud or TK cross down
            exit_signal = price_below_cloud or tk_cross_down
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price rises above cloud or TK cross up
            exit_signal = price_above_cloud or tk_cross_up
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_v1"
timeframe = "6h"
leverage = 1.0