#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku TK Cross with 1d Cloud Filter and Volume Spike Confirmation
# Ichimoku TK Cross (Tenkan/Kijun) provides momentum signals. 1d Cloud acts as a strong support/resistance filter.
# Volume spike confirms institutional participation. Designed for 12-37 trades/year on 6h to minimize fee drag.
# Works in bull markets via long TK crosses above cloud and in bear markets via short TK crosses below cloud.

name = "6h_IchimokuTK_Cross_1dCloud_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for TK calculation - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components for 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Align Ichimoku components to primary timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    
    # Get 1d data for Cloud (Senkou Span A/B) calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku Cloud components for 1d
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_span_a = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high_1d + period52_low_1d) / 2
    
    # Align Cloud components to primary timeframe (use previous day's values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long conditions: TK cross bullish AND price above cloud AND volume spike
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1] and  # Bullish crossover
                close[i] > cloud_top and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross bearish AND price below cloud AND volume spike
            elif (tenkan_aligned[i] < kijun_aligned[i] and 
                  tenkan_aligned[i-1] >= kijun_aligned[i-1] and  # Bearish crossover
                  close[i] < cloud_bottom and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross bearish OR price drops below cloud
            if (tenkan_aligned[i] < kijun_aligned[i] and 
                tenkan_aligned[i-1] >= kijun_aligned[i-1]) or \
               close[i] < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross bullish OR price rises above cloud
            if (tenkan_aligned[i] > kijun_aligned[i] and 
                tenkan_aligned[i-1] <= kijun_aligned[i-1]) or \
               close[i] > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals