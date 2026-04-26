#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1wTrend_VolumeFilter
Hypothesis: Use Ichimoku cloud twist (Senkou Span A/B cross) from weekly timeframe as trend filter, with 6h price breaking above/below the cloud (Senkou Span A) for entries, confirmed by volume spike (>1.8x 20-period average). Weekly trend ensures alignment with major market direction, reducing whipsaws in bear markets. Weekly timeframe provides stable trend signal, while 6h captures medium-term momentum. Target: 15-25 trades/year.
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
    
    # Get weekly data for Ichimoku calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 52):  # Need at least 52 periods for weekly Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over 9 periods
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1w).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1w).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over 26 periods
    period_kijun = 26
    max_high_kijun = pd.Series(high_1w).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1w).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over 52 periods shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1w).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Cloud twist: Senkou Span A crossing above/below Senkou Span B
    # Twist up: Senkou Span A > Senkou Span B (bullish twist)
    # Twist down: Senkou Span A < Senkou Sen B (bearish twist)
    twist_up = senkou_span_a > senkou_span_b
    twist_down = senkou_span_a < senkou_span_b
    
    # Align Ichimoku components to 6h timeframe
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    twist_up_aligned = align_htf_to_ltf(prices, df_1w, twist_up.astype(float))
    twist_down_aligned = align_htf_to_ltf(prices, df_1w, twist_down.astype(float))
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku calculation (52 periods) + volume MA warmup
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(twist_up_aligned[i]) or np.isnan(twist_down_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter from cloud twist
        weekly_bullish = twist_up_aligned[i] > 0.5
        weekly_bearish = twist_down_aligned[i] > 0.5
        
        if position == 0:
            # Long: price above Senkou Span A + weekly bullish twist + volume spike
            long_signal = (close[i] > senkou_span_a_aligned[i]) and weekly_bullish and volume_spike[i]
            
            # Short: price below Senkou Span A + weekly bearish twist + volume spike
            short_signal = (close[i] < senkou_span_a_aligned[i]) and weekly_bearish and volume_spike[i]
            
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
            # Exit: price touches Senkou Span A OR weekly trend turns bearish
            if (close[i] < senkou_span_a_aligned[i] or not weekly_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Senkou Span A OR weekly trend turns bullish
            if (close[i] > senkou_span_a_aligned[i] or not weekly_bearish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0