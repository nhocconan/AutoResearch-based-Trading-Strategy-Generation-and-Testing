#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_Confirmation
Hypothesis: 6h Ichimoku TK cross with 1d cloud filter. Long when TK cross bullish and price above 1d cloud (bullish regime). Short when TK cross bearish and price below 1d cloud (bearish regime). The 1d cloud acts as a strong regime filter, reducing whipsaws in sideways markets. Ichimoku works well on 6h timeframe as it captures medium-term momentum while the daily cloud filters out counter-trend noise. Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Calculate 1d cloud (Senkou Span A and B) from 1d data
    # Tenkan-sen 1d: (9-period high + 9-period low) / 2
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    # Kijun-sen 1d: (26-period high + 26-period low) / 2
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    # Senkou Span A 1d: (Tenkan-sen 1d + Kijun-sen 1d) / 2
    senkou_span_a_1d = (tenkan_sen_1d + kijun_sen_1d) / 2
    
    # Senkou Span B 1d: (52-period high + 52-period low) / 2
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = (period52_high_1d + period52_low_1d) / 2
    
    # Align 1d cloud components to 6h timeframe
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # The cloud is between Senkou Span A and B
    # Top of cloud = max(Senkou Span A, Senkou Span B)
    # Bottom of cloud = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_span_a_1d_aligned, senkou_span_b_1d_aligned)
    
    # TK Cross signals
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (max of all periods: 52 for Senkou Span B)
    start_idx = 52 + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: TK cross bullish AND price above 1d cloud (bullish regime)
        if tk_cross_bullish[i] and close[i] > cloud_top[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross bearish AND price below 1d cloud (bearish regime)
        elif tk_cross_bearish[i] and close[i] < cloud_bottom[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross reverses OR price crosses cloud in opposite direction
        elif position == 1 and (tk_cross_bearish[i] or close[i] < cloud_bottom[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (tk_cross_bullish[i] or close[i] > cloud_top[i]):
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

name = "6h_Ichimoku_Cloud_Filter_1dTrend_Confirmation"
timeframe = "6h"
leverage = 1.0