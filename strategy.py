#!/usr/bin/env python3
# 6H_1W_Ichimoku_Trend_Follow
# Hypothesis: Ichimoku on weekly timeframe provides strong trend filter; price above/below cloud indicates trend direction.
# Entry on 6h when Tenkan-sen crosses Kijun-sen in direction of weekly trend, with price outside cloud for confirmation.
# Works in bull/bear by following weekly trend. Target: 12-30 trades/year per symbol.

name = "6H_1W_Ichimoku_Trend_Follow"
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
    
    # Get weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for proper calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Determine weekly trend: price above/below cloud
    # Cloud top is max of Senkou Span A and B, cloud bottom is min
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Bullish trend: price above cloud
    bullish_trend = close > cloud_top
    # Bearish trend: price below cloud
    bearish_trend = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for Ichimoku calculations
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Tenkan/Kijun cross signals
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        
        if position == 0:
            # Enter long: bullish trend + Tenkan crosses above Kijun + price above cloud
            if bullish_trend[i] and tk_cross_up and close[i] > cloud_top[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + Tenkan crosses below Kijun + price below cloud
            elif bearish_trend[i] and tk_cross_down and close[i] < cloud_bottom[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or Tenkan crosses below Kijun
            if bearish_trend[i] or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or Tenkan crosses above Kijun
            if bullish_trend[i] or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals