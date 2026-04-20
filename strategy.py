#!/usr/bin/env python3
# 6h_1d_Ichimoku_Cloud_Breakout_TenkanKijun
# Hypothesis: Use 1d Ichimoku cloud (Senkou Span A/B) as trend filter and 6h Tenkan/Kijun cross for entry.
# Tenkan-sen (9-period) crossing above Kijun-sen (26-period) when price is above cloud = long signal.
# Tenkan-sen crossing below Kijun-sen when price is below cloud = short signal.
# Cloud acts as dynamic support/resistance to filter false breakouts.
# Designed to work in both bull (trend-following with cloud) and bear (counter-trend at cloud edges) markets.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Cloud_Breakout_TenkanKijun"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # Need at least 26*2 for Ichimoku
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Ichimoku cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        close_val = prices['close'].iloc[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(tenkan_val) or np.isnan(kijun_val) or 
            np.isnan(span_a) or np.isnan(span_b)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom (Senkou Span A/B form the cloud)
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price is above cloud
            if tenkan_val > kijun_val and close_val > cloud_top:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price is below cloud
            elif tenkan_val < kijun_val and close_val < cloud_bottom:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud
            if tenkan_val < kijun_val or close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud
            if tenkan_val > kijun_val or close_val > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals