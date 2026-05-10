#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Filter_v2
# Hypothesis: Ichimoku cloud with 1d Tenkan/Kijun cross + Kumo twist filter on 6h timeframe.
# Uses 1d Ichimoku components for trend direction and Kumo twist for trend strength.
# Entry: Price above/below cloud + TK cross aligned with Kumo twist direction.
# Exit: Price enters cloud or TK cross reverses.
# Designed for 6h timeframe with 1d Ichimoku filter to reduce whipsaws in sideways markets.
# Target: 20-50 trades/year (80-200 total over 4 years) with controlled risk.

name = "6h_Ichimoku_Cloud_Trend_Filter_v2"
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
    
    # 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = np.full_like(high_1d, np.nan)
    period9_low = np.full_like(low_1d, np.nan)
    for i in range(8, len(high_1d)):
        period9_high[i] = np.max(high_1d[i-8:i+1])
        period9_low[i] = np.min(low_1d[i-8:i+1])
    tenkan_1d = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = np.full_like(high_1d, np.nan)
    period26_low = np.full_like(low_1d, np.nan)
    for i in range(25, len(high_1d)):
        period26_high[i] = np.max(high_1d[i-25:i+1])
        period26_low[i] = np.min(low_1d[i-25:i+1])
    kijun_1d = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period52_high = np.full_like(high_1d, np.nan)
    period52_low = np.full_like(low_1d, np.nan)
    for i in range(51, len(high_1d)):
        period52_high[i] = np.max(high_1d[i-51:i+1])
        period52_low[i] = np.min(low_1d[i-51:i+1])
    senkou_b_1d = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Calculate Kumo twist (Senkou A - Senkou B) for trend strength
    # Positive: bullish twist, Negative: bearish twist
    kumo_twist = senkou_a_aligned - senkou_b_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for Ichimoku calculations
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or \
           np.isnan(kumo_twist[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud boundaries: upper band = max(Senkou A, Senkou B), lower band = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long: Price above cloud + TK cross bullish + Kumo twist bullish (positive)
            if (close[i] > cloud_top and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                kumo_twist[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + TK cross bearish + Kumo twist bearish (negative)
            elif (close[i] < cloud_bottom and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  kumo_twist[i] < 0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price enters cloud or TK cross turns bearish
            if close[i] < cloud_top or tenkan_aligned[i] <= kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price enters cloud or TK cross turns bullish
            if close[i] > cloud_bottom or tenkan_aligned[i] >= kijun_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals