#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Weekly Pivot Direction + Volume Confirmation
- Uses Ichimoku on 1d HTF for trend (Tenkan/Kijun cross + price vs cloud)
- Weekly pivot from 1w HTF for directional bias (price above/below weekly pivot)
- Volume confirmation: current volume > 1.5 * 20-bar median
- Long when: bullish Ichimoku (Tenkan>Kijun, price>cloud) AND price>weekly pivot AND volume confirmation
- Short when: bearish Ichimoku (Tenkan<Kijun, price<cloud) AND price<weekly pivot AND volume confirmation
- Exit on opposite Ichimoku cross or weekly pivot cross
- Designed for 6h timeframe targeting 50-150 total trades over 4 years (12-37/year)
- Ichimoku provides trend strength and support/resistance via cloud
- Weekly pivot adds higher timeframe structure to avoid counter-trend trades
- Volume confirmation filters low-volatility breakouts
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
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Ichimoku
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
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Get 1w data ONCE before loop for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot (standard: (H+L+C)/3)
    # Use previous week's data for pivot calculation
    week_high = df_1w['high'].values
    week_low = df_1w['low'].values
    week_close = df_1w['close'].values
    
    weekly_pivot = (week_high + week_low + week_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_median[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Ichimoku bullish: Tenkan > Kijun AND price above cloud
        ichimoku_bullish = (tenkan_aligned[i] > kijun_aligned[i]) and (close[i] > cloud_top)
        # Ichimoku bearish: Tenkan < Kijun AND price below cloud
        ichimoku_bearish = (tenkan_aligned[i] < kijun_aligned[i]) and (close[i] < cloud_bottom)
        
        if position == 0:
            # Long: bullish Ichimoku AND price > weekly pivot AND volume confirmation
            if ichimoku_bullish and (close[i] > weekly_pivot_aligned[i]) and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Ichimoku AND price < weekly pivot AND volume confirmation
            elif ichimoku_bearish and (close[i] < weekly_pivot_aligned[i]) and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish Ichimoku cross OR price crosses below weekly pivot
            if ichimoku_bearish or (close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish Ichimoku cross OR price crosses above weekly pivot
            if ichimoku_bullish or (close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0