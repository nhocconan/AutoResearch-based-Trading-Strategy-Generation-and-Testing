#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud Breakout with 1d Weekly Pivot Filter
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) on 6h for trend and momentum.
# Weekly pivot levels from 1d data filter trades: only take longs above weekly pivot, shorts below.
# Volume confirmation ensures breakout strength. Works in bull (breakouts above cloud + pivot) and
# bear (breakouts below cloud + pivot) markets. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Ichimoku on 6h
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Load 1d data for weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot from previous week (using last 5 trading days approx)
    # Using 5-day high/low/close for weekly pivot
    week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly Pivot Point = (Week High + Week Low + Week Close) / 3
    weekly_pivot = (week_high + week_low + week_close) / 3
    
    # Align Ichimoku and weekly pivot to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), senkou_span_b)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Long conditions:
        # 1. Tenkan-sen crosses above Kijun-sen (bullish momentum)
        # 2. Price above cloud (bullish trend)
        # 3. Price above weekly pivot (bullish bias from higher timeframe)
        # 4. Volume confirmation (above 20-period median)
        tenkan_cross = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        price_above_cloud = close[i] > cloud_top
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        volume_filter = volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1])
        
        if tenkan_cross and price_above_cloud and price_above_pivot and volume_filter and position <= 0:
            position = 1
            signals[i] = base_size
        
        # Short conditions:
        # 1. Tenkan-sen crosses below Kijun-sen (bearish momentum)
        # 2. Price below cloud (bearish trend)
        # 3. Price below weekly pivot (bearish bias from higher timeframe)
        # 4. Volume confirmation
        elif (tenkan_sen_aligned[i] < kijun_sen_aligned[i] and
              close[i] < cloud_bottom and
              close[i] < weekly_pivot_aligned[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or price returns to cloud
        elif position == 1 and (tenkan_sen_aligned[i] < kijun_sen_aligned[i] or close[i] < cloud_bottom):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (tenkan_sen_aligned[i] > kijun_sen_aligned[i] or close[i] > cloud_top):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyPivot"
timeframe = "6h"
leverage = 1.0