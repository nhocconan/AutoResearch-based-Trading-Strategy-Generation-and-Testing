#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Filter
# Hypothesis: The Ichimoku cloud from the daily timeframe acts as a dynamic support/resistance zone.
# Price above the cloud indicates bullish trend; price below indicates bearish trend.
# We enter long when Tenkan-sen crosses above Kijun-sen AND price is above the cloud,
# and enter short when Tenkan-sen crosses below Kijun-sen AND price is below the cloud.
# This strategy works in both bull and bear markets by following the higher-timeframe trend,
# and the Ichimoku cloud provides natural trend filtering with low trade frequency.

name = "6h_Ichimoku_Cloud_Trend_Filter"
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough history for Senkou Span B
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or \
           np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1] and
                close[i] > cloud_top):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and
                  close[i] < cloud_bottom):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below cloud OR Tenkan crosses below Kijun
            if close[i] < cloud_bottom or \
               (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above cloud OR Tenkan crosses above Kijun
            if close[i] > cloud_top or \
               (tenkan_sen_aligned[i] > kijun_sen_aligned[i] and 
                tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals