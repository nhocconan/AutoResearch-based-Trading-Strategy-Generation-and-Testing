#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + TK Cross + Weekly Pivot Filter
# Uses Ichimoku system on 6h for trend detection with TK cross for entry timing.
# Weekly pivot (from 1w data) provides directional bias: only take longs above weekly pivot,
# shorts below weekly pivot. Combines momentum, trend, and institutional levels.
# Designed to work in both bull and bear markets by requiring alignment with weekly structure.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Ichimoku
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1w data for weekly pivot (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal generation, we don't need Chikou Span directly
    
    # Calculate weekly pivot points from 1w data
    # Standard pivot: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    # Resistance 1: (2 * P) - L
    weekly_r1 = (2 * weekly_pivot) - low_1w
    # Support 1: (2 * P) - H
    weekly_s1 = (2 * weekly_pivot) - high_1w
    
    # Align Ichimoku components to 6h timeframe (already aligned since from df_6h)
    # Align weekly pivot data to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(52, n):  # Start after 52-period for Senkou Span B
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = np.minimum(senkou_span_a[i], senkou_span_b[i])
        
        # Long entry: TK cross bullish + price above cloud + above weekly pivot
        if (tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i-1] <= kijun_sen[i-1] and
            close[i] > cloud_top and
            close[i] > weekly_pivot_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: TK cross bearish + price below cloud + below weekly pivot
        elif (tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i-1] >= kijun_sen[i-1] and
              close[i] < cloud_bottom and
              close[i] < weekly_pivot_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: TK cross in opposite direction or price crosses weekly pivot
        elif position == 1 and (tenkan_sen[i] < kijun_sen[i] or close[i] < weekly_pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen[i] > kijun_sen[i] or close[i] > weekly_pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_WeeklyPivot"
timeframe = "6h"
leverage = 1.0