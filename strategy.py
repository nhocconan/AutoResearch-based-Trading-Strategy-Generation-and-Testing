#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud + Weekly Pivot Direction + Volume Confirmation
- Ichimoku (1d): Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (26, 52 displacement 26)
- Long: Price > Cloud AND Tenkan > Kijun AND Weekly bullish bias (price > weekly pivot) AND volume > 1.5x 20-period avg
- Short: Price < Cloud AND Tenkan < Kijun AND Weekly bearish bias (price < weekly pivot) AND volume > 1.5x 20-period avg
- Exit: Opposite Ichimoku alignment (Tenkan/Kijun cross) OR price re-enters cloud
- Uses Ichimoku for trend structure and momentum, weekly pivot for HTF bias, volume for confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (price above cloud + bullish alignment) and bear markets (price below cloud + bearish alignment)
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
              pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    tenkan_values = tenkan.values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
             pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    kijun_values = kijun.values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_a_values = senkou_a.values
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    senkou_b = senkou_b.shift(26)
    senkou_b_values = senkou_b.values
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_values)
    
    # Calculate 1w pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot: (Prior week high + low + close) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_values = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need 52+26=78 for Senkou B, 26 for Tenkan/Kijun, 20 for volume MA
    start_idx = max(78, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or
            np.isnan(senkou_b_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Ichimoku Cloud: Price above/below cloud
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan/Kijun cross
        tenkan_gt_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_lt_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Weekly bias
        weekly_bullish = close[i] > weekly_pivot_aligned[i]
        weekly_bearish = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: Price > Cloud AND Tenkan > Kijun AND Weekly bullish AND volume confirmation
            if (price_above_cloud and 
                tenkan_gt_kijun and 
                weekly_bullish and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price < Cloud AND Tenkan < Kijun AND Weekly bearish AND volume confirmation
            elif (price_below_cloud and 
                  tenkan_lt_kijun and 
                  weekly_bearish and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan <= Kijun OR price re-enters cloud (price < cloud top)
            if not tenkan_gt_kijun or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan >= Kijun OR price re-enters cloud (price > cloud bottom)
            if not tenkan_lt_kijun or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_WeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0