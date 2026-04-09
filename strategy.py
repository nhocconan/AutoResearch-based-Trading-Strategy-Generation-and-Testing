#!/usr/bin/env python3
# 6h_ichimoku_cloud_12h_v1
# Hypothesis: 6h Ichimoku Cloud with 12h trend filter for high-probability trend continuation.
# Uses 6h timeframe to balance trade frequency (target 12-37/year). Ichimoku provides
# dynamic support/resistance (cloud) and momentum (TK cross). 12h timeframe acts as
# trend filter: only take long signals when 12h is bullish (price above cloud) and
# short signals when 12h is bearish (price below cloud). Volume confirmation filters
# low-participation breakouts. Works in bull/bear markets: Ichimoku adapts to volatility,
# 12h filter avoids counter-trend trades during ranging, volume confirms institutional interest.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_12h_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    if len(high) < kijun:
        return (np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float),
                np.full_like(high, np.nan, dtype=float))
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max().values
    period9_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max().values
    period26_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max().values
    period52_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_spans = np.roll(close, -kijun)
    chikou_spans[-kijun:] = np.nan  # Will be handled by alignment
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou_spans

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku on 6h
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high_6h, low_6h, close_6h)
    
    # Align Ichimoku components to 6h timeframe (completed 6h candle only)
    tenkan_6h_aligned = align_htf_to_ltf(prices, df_6h, tenkan_6h)
    kijun_6h_aligned = align_htf_to_ltf(prices, df_6h, kijun_6h)
    senkou_a_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_a_6h)
    senkou_b_6h_aligned = align_htf_to_ltf(prices, df_6h, senkou_b_6h)
    chikou_6h_aligned = align_htf_to_ltf(prices, df_6h, chikou_6h)
    
    # Cloud: Senkou Span A and B
    upper_cloud_6h = np.maximum(senkou_a_6h_aligned, senkou_b_6h_aligned)
    lower_cloud_6h = np.minimum(senkou_a_6h_aligned, senkou_b_6h_aligned)
    
    # Get 12h HTF data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Ichimoku for trend filter (simplified: just cloud)
    tenkan_12h, kijun_12h, senkou_a_12h, senkou_b_12h, _ = calculate_ichimoku(high_12h, low_12h, close_12h)
    
    # Align 12h Ichimoku to 6h timeframe
    tenkan_12h_aligned = align_htf_to_ltf(prices, df_12h, tenkan_12h)
    kijun_12h_aligned = align_htf_to_ltf(prices, df_12h, kijun_12h)
    senkou_a_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_a_12h)
    senkou_b_12h_aligned = align_htf_to_ltf(prices, df_12h, senkou_b_12h)
    
    # 12h Cloud
    upper_cloud_12h = np.maximum(senkou_a_12h_aligned, senkou_b_12h_aligned)
    lower_cloud_12h = np.minimum(senkou_a_12h_aligned, senkou_b_12h_aligned)
    
    # 12h Trend: price above/below cloud
    # For 12h trend, we need to align the 12h close to 6h as well
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    trend_bull_12h = close_12h_aligned > upper_cloud_12h
    trend_bear_12h = close_12h_aligned < lower_cloud_12h
    
    # Volume confirmation: 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or
            np.isnan(upper_cloud_6h[i]) or np.isnan(lower_cloud_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h cloud OR TK cross turns bearish
            if (close[i] < lower_cloud_6h[i]) or (tenkan_6h_aligned[i] < kijun_6h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h cloud OR TK cross turns bullish
            if (close[i] > upper_cloud_6h[i]) or (tenkan_6h_aligned[i] > kijun_6h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above 6h cloud, TK bullish, 12h bullish trend, volume spike
            if (close[i] > upper_cloud_6h[i]) and \
               (tenkan_6h_aligned[i] > kijun_6h_aligned[i]) and \
               trend_bull_12h[i] and \
               vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price below 6h cloud, TK bearish, 12h bearish trend, volume spike
            elif (close[i] < lower_cloud_6h[i]) and \
                 (tenkan_6h_aligned[i] < kijun_6h_aligned[i]) and \
                 trend_bear_12h[i] and \
                 vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals