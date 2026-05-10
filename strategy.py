#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Volume
Hypothesis: Use Ichimoku Cloud from 1d timeframe to define trend and support/resistance, with price breaking above/below cloud for entry, filtered by volume and TK cross on 6h. The Ichimoku Cloud provides dynamic support/resistance and trend direction, while the TK cross adds momentum confirmation. Volume ensures breakout strength. Designed for 12-30 trades/year on 6h, works in both bull and bear via trend filter.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku Cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Ichimoku Cloud components (using standard periods: 9, 26, 52)
    high_9 = df_1d['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_1d['low'].rolling(window=9, min_periods=9).min()
    high_26 = df_1d['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_1d['low'].rolling(window=26, min_periods=26).min()
    high_52 = df_1d['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_1d['low'].rolling(window=52, min_periods=52).min()
    
    tenkan_sen = (high_9 + low_9) / 2  # Conversion Line
    kijun_sen = (high_26 + low_26) / 2  # Base Line
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)  # Leading Span A
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)  # Leading Span B
    
    # Current cloud boundaries (Senkou Span A/B)
    senkou_a = senkou_span_a.values
    senkou_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe (wait for 1d bar to close)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52 periods for Senkou B) + TK cross + volume EMA
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        lower_cloud = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Bullish trend: price above cloud
        # Bearish trend: price below cloud
        
        if position == 0:
            # Long: price breaks above cloud AND bullish TK cross (Tenkan > Kijun) with volume
            if close[i] > upper_cloud and tenkan_sen_aligned[i] > kijun_sen_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud AND bearish TK cross (Tenkan < Kijun) with volume
            elif close[i] < lower_cloud and tenkan_sen_aligned[i] < kijun_sen_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below cloud OR bearish TK cross
            if close[i] < lower_cloud or tenkan_sen_aligned[i] < kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above cloud OR bullish TK cross
            if close[i] > upper_cloud or tenkan_sen_aligned[i] > kijun_sen_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals