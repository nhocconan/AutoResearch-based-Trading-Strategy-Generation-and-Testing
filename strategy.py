#!/usr/bin/env python3
# 6h_1dIchimoku_Cloud_TF_Breakout
# Uses daily Ichimoku cloud (Tenkan/Kijun/Senkou) with 6h price breakout and volume confirmation.
# Long when price breaks above Senkou Span A in uptrend (price > Kumo cloud top and Tenkan > Kijun), short when breaks below Senkou Span B in downtrend.
# Ichimoku components calculated on daily timeframe and aligned to 6h with proper look-ahead prevention.
# Designed for 6h timeframe to capture institutional trend continuation in both bull and bear markets.

name = "6h_1dIchimoku_Cloud_TF_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B (26*2)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Align Ichimoku components to 6h timeframe (no extra delay needed as these are concurrent indicators)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6-period volume average for spike detection (2x average)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (2.0 * vol_ma_6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period to prevent churn
    
    for i in range(52, n):  # Start after Senkou Span B calculation is valid
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Determine cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = min(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        if position == 0:
            # Long: price breaks above cloud top with bullish alignment (Tenkan > Kijun) and volume
            if close[i] > cloud_top and tenkan_sen_6h[i] > kijun_sen_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below cloud bottom with bearish alignment (Tenkan < Kijun) and volume
            elif close[i] < cloud_bottom and tenkan_sen_6h[i] < kijun_sen_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price returns below cloud bottom or Tenkan crosses below Kijun
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] < cloud_bottom or tenkan_sen_6h[i] < kijun_sen_6h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above cloud top or Tenkan crosses above Kijun
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry >= 3 and (close[i] > cloud_top or tenkan_sen_6h[i] > kijun_sen_6h[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals