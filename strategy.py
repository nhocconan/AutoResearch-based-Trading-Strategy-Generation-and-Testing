#!/usr/bin/env python3
# 6h_Ichimoku_CloudFilter_1dTrend_Volume
# Hypothesis: 6h chart strategy using Ichimoku TK cross with cloud filter from 1d timeframe and volume confirmation.
# Long when TK crosses above, price is above cloud (bullish), and volume > 1.5x average.
# Short when TK crosses below, price is below cloud (bearish), and volume > 1.5x average.
# Uses daily Ichimoku for trend context to avoid counter-trend trades in both bull and bear markets.
# Target: 20-40 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "6h"
name = "6h_Ichimoku_CloudFilter_1dTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume spike detection: 1.5x average volume (4-period = 1 day on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need full Ichimoku data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, volume confirmation
            tk_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            tk_cross_prev = tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross and tk_cross_prev and price_above_cloud and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK cross bearish, price below cloud, volume confirmation
            elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
                  tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1] and
                  close[i] < cloud_bottom and
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TK cross bearish or price drops below cloud
            tk_cross_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
            tk_cross_bearish_prev = tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
            price_below_cloud = close[i] < cloud_bottom
            
            if tk_cross_bearish and tk_cross_bearish_prev or price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TK cross bullish or price rises above cloud
            tk_cross_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
            tk_cross_bullish_prev = tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
            price_above_cloud = close[i] > cloud_top
            
            if tk_cross_bullish and tk_cross_bullish_prev or price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals