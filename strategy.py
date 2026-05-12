#!/usr/bin/env python3
"""
6h_Ichimoku_KumoBreakout_1wTrend_Volume
Hypothesis: Trade Ichimoku Kumo (cloud) breakouts on 6h timeframe when aligned with 1-week trend (price above/below weekly Kumo) and confirmed by volume spike. The Ichimoku cloud acts as dynamic support/resistance, while the weekly trend filter ensures we trade with the higher timeframe momentum. Volume spike confirms institutional participation. This combination should work in both bull and bear markets by capturing major trend continuations after consolidation periods.
Timeframe: 6h
"""

name = "6h_Ichimoku_KumoBreakout_1wTrend_Volume"
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

    # Get weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for Ichimoku
        return np.zeros(n)

    # Calculate weekly Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for trend)
    
    # The Kumo (cloud) is between Senkou Span A and Senkou Span B
    # For trend filter: price above cloud = bullish, price below cloud = bearish
    # We need to shift Senkou spans forward by 26 periods to align with current price
    # But since we're using weekly data for trend filter on 6h chart, we align directly
    
    # Align weekly Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Get daily data for 6h Ichimoku (entry signal) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
        
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily Ichimoku components for 6h entry signals
    period9_high_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_span_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    period52_high_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    # Align daily Ichimoku to 6h
    tenkan_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen_1d)
    kijun_sen_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen_1d)
    senkou_span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d)
    senkou_span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d)
    
    # Volume spike: current > 2.0x average of last 4 bars (approx 1 day on 6h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after warmup
        if (np.isnan(tenkan_sen_1d_aligned[i]) or np.isnan(kijun_sen_1d_aligned[i]) or
            np.isnan(senkou_span_a_1d_aligned[i]) or np.isnan(senkou_span_b_1d_aligned[i]) or
            np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Kumo (cloud) boundaries for 6d (senkou spans)
        upper_kumo = np.maximum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        lower_kumo = np.minimum(senkou_span_a_1d_aligned[i], senkou_span_b_1d_aligned[i])
        
        # Weekly trend filter: price relative to weekly Kumo
        weekly_upper_kumo = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        weekly_lower_kumo = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price_above_weekly_kumo = close[i] > weekly_upper_kumo
        price_below_weekly_kumo = close[i] < weekly_lower_kumo

        if position == 0:
            # LONG: price breaks above 6h Kumo + price above weekly Kumo + volume spike
            if (close[i] > upper_kumo and 
                price_above_weekly_kumo and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below 6h Kumo + price below weekly Kumo + volume spike
            elif (close[i] < lower_kumo and 
                  price_below_weekly_kumo and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below 6h Kumo OR Tenkan-Kijun cross down
            tenkan_kijun_cross_down = tenkan_sen_1d_aligned[i] < kijun_sen_1d_aligned[i]
            if close[i] < lower_kumo or tenkan_kijun_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above 6h Kumo OR Tenkan-Kijun cross up
            tenkan_kijun_cross_up = tenkan_sen_1d_aligned[i] > kijun_sen_1d_aligned[i]
            if close[i] > upper_kumo or tenkan_kijun_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals