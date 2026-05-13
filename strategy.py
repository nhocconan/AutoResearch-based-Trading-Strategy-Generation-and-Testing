#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_Force
Hypothesis: Combines Ichimoku cloud from 1d timeframe for trend direction and breakouts from 6h price action with volume confirmation. 
In bullish regime (price above 1d cloud), look for 6h breaks above Tenkan/Kijun resistance with volume spike. 
In bearish regime (price below 1d cloud), look for 6h breaks below cloud support with volume spike.
Ichimoku provides multi-timeframe context while 6h captures momentum shifts. Designed to work in both bull and bear markets by adapting to trend regime.
"""

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_Force"
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
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if Ichimoku data not available
        if np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above cloud (bullish regime) and breaks above Tenkan/Kijun resistance with volume
            if close[i] > cloud_top[i] and \
               high[i] > max(tenkan_sen_aligned[i], kijun_sen_aligned[i]) and \
               volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud (bearish regime) and breaks below cloud support with volume
            elif close[i] < cloud_bottom[i] and \
                 low[i] < cloud_bottom[i] and \
                 volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below cloud or Tenkan/Kijun support
            if close[i] < cloud_bottom[i] or low[i] < min(tenkan_sen_aligned[i], kijun_sen_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above cloud
            if close[i] > cloud_top[i] or high[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals