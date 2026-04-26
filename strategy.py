#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, use Ichimoku cloud twist (Senkou Span A/B cross) from 1d for regime detection, combined with price breaking above/below the cloud for entries. Filter by 1d trend (price >/ < Kumo midpoint) and volume spike (>1.8x 20-period average). Enter long when price breaks above cloud with bullish twist, 1d uptrend, and volume spike. Enter short when price breaks below cloud with bearish twist, 1d downtrend, and volume filter. Uses discrete position size 0.25. Designed for 12-30 trades/year on 6h by requiring multiple confluence factors, reducing overtrading while capturing structural breaks in both bull and bear markets.
"""

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
    
    # Get 1d data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Kumo twist: when Senkou Span A crosses above/below Senkou Span B
    # Bullish twist: Senkou Span A > Senkou Span B
    # Bearish twist: Senkou Span A < Senkou Span B
    bullish_twist = senkou_span_a > senkou_span_b
    bearish_twist = senkou_span_a < senkou_span_b
    
    # Align Ichimoku components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    # Kumo midpoint for trend filter: (Senkou Span A + Senkou Span B) / 2
    kumo_midpoint = (senkou_span_a + senkou_span_b) / 2
    kumo_midpoint_aligned = align_htf_to_ltf(prices, df_1d, kumo_midpoint)
    
    # 1d trend filter: price relative to Kumo midpoint
    trend_1d_uptrend = close > kumo_midpoint_aligned
    trend_1d_downtrend = close < kumo_midpoint_aligned
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 52 periods for Senkou Span B, 20 for volume MA
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(bullish_twist_aligned[i]) or np.isnan(bearish_twist_aligned[i]) or
            np.isnan(kumo_midpoint_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Price relative to cloud
        price_above_cloud = close[i] > np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_below_cloud = close[i] < np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: price breaks above cloud + bullish twist + 1d uptrend + volume spike
            long_signal = (price_above_cloud and 
                          bullish_twist_aligned[i] > 0.5 and 
                          trend_1d_uptrend[i] and 
                          volume_spike[i])
            
            # Short: price breaks below cloud + bearish twist + 1d downtrend + volume spike
            short_signal = (price_below_cloud and 
                           bearish_twist_aligned[i] > 0.5 and 
                           trend_1d_downtrend[i] and 
                           volume_spike[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below cloud OR twist turns bearish
            if (price_below_cloud or bullish_twist_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud OR twist turns bullish
            if (price_above_cloud or bearish_twist_aligned[i] <= 0.5):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0