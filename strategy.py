#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Ichimoku_Cloud_TK_Cross_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data once for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window, min_periods=window).min().values
    
    tenkan_sen = (rolling_max(high_1d, 9) + rolling_min(low_1d, 9)) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (rolling_max(high_1d, 26) + rolling_min(low_1d, 26)) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((rolling_max(high_1d, 52) + rolling_min(low_1d, 52)) / 2)
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = close_1d  # Will be used without shift for current price comparison
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_aligned = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # warmup for Ichimoku (max period 52)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(chikou_span_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku Cloud: top and bottom of the cloud
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long entry: TK Cross bullish (Tenkan > Kijun) AND price above cloud
            tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            price_above_cloud = close[i] > cloud_top
            
            # Short entry: TK Cross bearish (Tenkan < Kijun) AND price below cloud
            tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_bullish and price_above_cloud:
                signals[i] = 0.25
                position = 1
            elif tk_bearish and price_below_cloud:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK Cross bearish OR price drops below cloud
            if (tenkan_sen_aligned[i] < kijun_sen_aligned[i]) or (close[i] < cloud_bottom):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK Cross bullish OR price rises above cloud
            if (tenkan_sen_aligned[i] > kijun_sen_aligned[i]) or (close[i] > cloud_top):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Cloud provides dynamic support/resistance while TK Cross signals momentum.
# Long when Tenkan crosses above Kijun (bullish momentum) AND price is above the cloud (bullish structure).
# Short when Tenkan crosses below Kijun (bearish momentum) AND price is below the cloud (bearish structure).
# Works in trending markets (cloud acts as dynamic S/R) and range-bound markets (TK cross signals reversals).
# Uses daily Ichimoku for higher timeframe context, executed on 6h for better entry timing.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.