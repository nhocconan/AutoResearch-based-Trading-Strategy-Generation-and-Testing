#!/usr/bin/env python3
# 6h_1d_ichimoku_cloud_v1
# Strategy: 6h Ichimoku with 1d cloud filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Ichimoku system provides dynamic support/resistance and momentum signals.
# Using 1d cloud as trend filter improves robustness in both bull/bear markets.
# Volume confirmation reduces false signals. Designed for 30-60 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 1d Ichimoku components
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
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h volume filter: current volume > 20-period average volume
    volume_series = pd.Series(volume)
    vol_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_span_a_6h[i]) or np.isnan(senkou_span_b_6h[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        cloud_bottom = np.minimum(senkou_span_a_6h[i], senkou_span_b_6h[i])
        
        # Bullish: price above cloud AND Tenkan > Kijun
        bullish_setup = (close[i] > cloud_top) and (tenkan_sen_6h[i] > kijun_sen_6h[i])
        
        # Bearish: price below cloud AND Tenkan < Kijun
        bearish_setup = (close[i] < cloud_bottom) and (tenkan_sen_6h[i] < kijun_sen_6h[i])
        
        # Entry logic: Ichimoku signal + volume filter
        if bullish_setup and volume_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_setup and volume_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price enters cloud or Tenkan/Kijun cross reverses
        elif position == 1 and (close[i] <= cloud_top or tenkan_sen_6h[i] <= kijun_sen_6h[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= cloud_bottom or tenkan_sen_6h[i] >= kijun_sen_6h[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals