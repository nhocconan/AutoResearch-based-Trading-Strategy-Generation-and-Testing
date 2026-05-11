#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_Cloud_Filter_1d
Hypothesis: Use Ichimoku Cloud from 1d timeframe with TK Cross (Tenkan/Kijun) on 6h for entry signals.
In bull markets: price above cloud + TK cross up = long. In bear markets: price below cloud + TK cross down = short.
The cloud acts as dynamic support/resistance and trend filter, reducing whipsaws. TK cross provides timely entries.
Target: 15-30 trades per year on 6h timeframe.
"""

name = "6h_Ichimoku_TK_Cross_Cloud_Filter_1d"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1D Data for Ichimoku Cloud ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Chikou Span (Lagging Span): not used for signals
    
    # Align 1D Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Calculate 6h TK Cross
    # Tenkan/Kijun cross on 6h
    tenkan_6h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_6h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    
    tk_cross = tenkan_6h - kijun_6h
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = 0  # avoid false signal on first bar
    
    # Bullish TK cross: tenkan crosses above kijun
    tk_cross_up = (tk_cross > 0) & (tk_cross_prev <= 0)
    # Bearish TK cross: tenkan crosses below kijun
    tk_cross_down = (tk_cross < 0) & (tk_cross_prev >= 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud boundaries
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: price above cloud AND bullish TK cross on 6h
            if close[i] > cloud_top and tk_cross_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud AND bearish TK cross on 6h
            elif close[i] < cloud_bottom and tk_cross_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below cloud OR bearish TK cross
            if close[i] < cloud_bottom or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above cloud OR bullish TK cross
            if close[i] > cloud_top or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals