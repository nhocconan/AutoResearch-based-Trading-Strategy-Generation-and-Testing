#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, enter long when price breaks above Kumo (cloud) AND weekly trend is bullish (price > weekly Kijun-sen) AND volume > 2.0x 20-period average. Enter short when price breaks below Kumo AND weekly trend is bearish (price < weekly Kijun-sen) AND volume spike. Uses Ichimoku cloud from 1d for dynamic support/resistance, weekly Kijun-sen for higher timeframe trend filter, and volume confirmation for institutional participation. Designed for moderate trade frequency (12-30/year) to avoid fee drag while capturing strong trends in both bull and bear markets via cloud breakouts with trend alignment.
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
    
    # Get 1d data for Ichimoku cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    tenkan_sen = tenkan_sen.values
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    kijun_sen = kijun_sen.values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 displaced 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods displaced 26 periods ahead
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    senkou_span_b = senkou_span_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Get 1w data for weekly Kijun-sen trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Kijun-sen: (highest high + lowest low)/2 for past 26 weeks
    weekly_kijun_sen = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                        pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    weekly_kijun_sen = weekly_kijun_sen.values
    
    # Align weekly Kijun-sen to 6h timeframe
    weekly_kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, weekly_kijun_sen)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Cloud top = max(Senkou Span A, Senkou Span B)
    # Cloud bottom = min(Senkou Span A, Senkou Span B)
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52), volume MA warmup (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(weekly_kijun_sen_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        weekly_trend_bullish = close[i] > weekly_kijun_sen_aligned[i]
        weekly_trend_bearish = close[i] < weekly_kijun_sen_aligned[i]
        
        # Kumo breakout conditions
        breakout_above_cloud = close[i] > cloud_top[i]
        breakout_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long: price above cloud + weekly bullish trend + volume spike
            long_signal = breakout_above_cloud and weekly_trend_bullish and volume_spike[i]
            
            # Short: price below cloud + weekly bearish trend + volume spike
            short_signal = breakout_below_cloud and weekly_trend_bearish and volume_spike[i]
            
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
            # Exit: price breaks below cloud bottom OR weekly trend turns bearish
            if close[i] < cloud_bottom[i] or not weekly_trend_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above cloud top OR weekly trend turns bullish
            if close[i] > cloud_top[i] or not weekly_trend_bearish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0