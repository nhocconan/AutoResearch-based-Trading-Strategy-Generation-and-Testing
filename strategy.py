#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_TK_Cross_v1
Hypothesis: 6h Ichimoku Cloud (from 1d) as trend filter + TK Cross (Tenkan/Kijun) for entry timing.
Ichimoku Cloud provides dynamic support/resistance that adapts to volatility and trend strength.
TK Cross gives timely entries aligned with the cloud's bias.
Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
Uses discrete sizing (0.25) to manage drawdown and avoid overtrading. Works in both bull and bear markets
by only taking trades in the direction of the cloud (price above cloud = long bias, below = short bias).
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
    
    # Get 1d data for Ichimoku Cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                 pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_1d = tenkan_1d.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_1d = kijun_1d.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                        pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b_1d = senkou_span_b_1d.values
    
    # Align Ichimoku components to 6h timeframe (displaced forward by 26 periods)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d, additional_delay_bars=displacement)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d, additional_delay_bars=displacement)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Ichimoku calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # TK Cross signals
            tk_cross_up = (tenkan_1d_aligned[i] > kijun_1d_aligned[i]) and (tenkan_1d_aligned[i-1] <= kijun_1d_aligned[i-1])
            tk_cross_down = (tenkan_1d_aligned[i] < kijun_1d_aligned[i]) and (tenkan_1d_aligned[i-1] >= kijun_1d_aligned[i-1])
            
            # Long: TK cross up AND price above cloud (bullish bias)
            # Short: TK cross down AND price below cloud (bearish bias)
            long_signal = tk_cross_up and (close[i] > cloud_top)
            short_signal = tk_cross_down and (close[i] < cloud_bottom)
            
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
            # Exit when price falls below cloud (trend invalidation)
            exit_signal = close[i] < cloud_bottom
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price rises above cloud (trend invalidation)
            exit_signal = close[i] > cloud_top
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_TK_Cross_v1"
timeframe = "6h"
leverage = 1.0