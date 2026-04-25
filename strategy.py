#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross with 1d Weekly Pivot Filter
Hypothesis: Ichimoku TK cross (Tenkan/Kijun) provides momentum signals,
while price above/below cloud (from 1d HTF) filters for trend alignment.
Weekly pivot direction from 1d data adds regime filter to avoid counter-trend
whipsaws. Works in both bull/bear markets by only trading in direction of
weekly pivot and cloud trend. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku and weekly pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Weekly pivot from 1d data (using prior week's OHLC)
    # Group into weeks: week_start = Monday 00:00 UTC
    df_1d_copy = df_1d.copy()
    df_1d_copy['week_start'] = df_1d_copy.index.to_series().dt.to_period('W').dt.start_time
    weekly = df_1d_copy.groupby('week_start').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(weekly) < 2:
        return np.zeros(n)
    
    weekly_high = weekly['high'].values[:-1]  # Exclude current incomplete week
    weekly_low = weekly['low'].values[:-1]
    weekly_close = weekly['close'].values[:-1]
    
    # Weekly pivot point: (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Weekly R1: (2*P) - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: (2*P) - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align all 1d indicators to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], weekly_pivot, additional_delay_bars=0)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], weekly_r1, additional_delay_bars=0)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d.iloc[:-1], weekly_s1, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Ichimoku warmup (52 periods for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        tenkan_val = tenkan_aligned[i]
        kijun_val = kijun_aligned[i]
        senkou_a_val = senkou_a_aligned[i]
        senkou_b_val = senkou_b_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # TK cross
        tk_cross_above = tenkan_val > kijun_val
        tk_cross_below = tenkan_val < kijun_val
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Weekly pivot bias
        weekly_bullish = weekly_pivot_val > weekly_s1_val and weekly_pivot_val < weekly_r1_val
        weekly_bullish_strong = curr_close > weekly_pivot_val
        weekly_bearish_strong = curr_close < weekly_pivot_val
        
        if position == 0:
            # Long: TK cross bullish, price above cloud, weekly bias bullish
            long_entry = tk_cross_above and price_above_cloud and weekly_bullish_strong
            # Short: TK cross bearish, price below cloud, weekly bias bearish
            short_entry = tk_cross_below and price_below_cloud and weekly_bearish_strong
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: TK cross bearish OR price falls below cloud
            if tk_cross_below or not price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross bullish OR price rises above cloud
            if tk_cross_above or not price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_WeeklyPivot"
timeframe = "6h"
leverage = 1.0