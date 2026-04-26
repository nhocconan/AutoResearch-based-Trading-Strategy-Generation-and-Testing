#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_WeeklyFilter
Hypothesis: 6h Ichimoku cloud breakout with 1d trend filter (TK cross) and 1w trend filter (price vs weekly cloud).
Enters long when price breaks above Ichimoku cloud with bullish 1d TK cross and price above weekly cloud.
Enters short when price breaks below Ichimoku cloud with bearish 1d TK cross and price below weekly cloud.
Exits when price re-enters the cloud or TK cross reverses.
Ichimoku components calculated on 6h data, 1d TK cross for trend alignment, 1w cloud for regime filter.
Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.25).
Works in bull/bear by aligning with multiple timeframe trends to avoid counter-trend trades.
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
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): close plotted 26 periods behind (not used for signals)
    
    # Get 1d data for TK cross trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate 1d Tenkan and Kijun for TK cross
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    # TK cross: 1 if bullish (Tenkan > Kijun), -1 if bearish (Tenkan < Kijun), 0 otherwise
    tk_cross_1d = np.where(tenkan_1d > kijun_1d, 1, np.where(tenkan_1d < kijun_1d, -1, 0))
    tk_cross_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_1d)
    
    # Get 1w data for weekly cloud regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_tenkan_1w + low_tenkan_1w) / 2
    
    high_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_kijun_1w + low_kijun_1w) / 2
    
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    high_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_senkou_b_1w + low_senkou_b_1w) / 2
    
    # Weekly cloud top and bottom
    weekly_cloud_top = np.maximum(senkou_a_1w, senkou_b_1w)
    weekly_cloud_bottom = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align all HTF indicators to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, prices, tenkan)  # already 6h
    kijun_aligned = align_htf_to_ltf(prices, prices, kijun)    # already 6h
    senkou_a_aligned = align_htf_to_ltf(prices, prices, senkou_a)  # already 6h
    senkou_b_aligned = align_htf_to_ltf(prices, prices, senkou_b)  # already 6h
    
    weekly_cloud_top_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_top)
    weekly_cloud_bottom_aligned = align_htf_to_ltf(prices, df_1w, weekly_cloud_bottom)
    
    # Cloud top and bottom (for 6h cloud)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 26 for TK cross alignment)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(weekly_cloud_top_aligned[i]) or 
            np.isnan(weekly_cloud_bottom_aligned[i]) or np.isnan(tk_cross_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above cloud with bullish 1d TK cross and price above weekly cloud
            if (close[i] > cloud_top[i] and 
                tk_cross_aligned[i] == 1 and 
                close[i] > weekly_cloud_top_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud with bearish 1d TK cross and price below weekly cloud
            elif (close[i] < cloud_bottom[i] and 
                  tk_cross_aligned[i] == -1 and 
                  close[i] < weekly_cloud_bottom_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price re-enters cloud OR TK cross turns bearish
            if (close[i] < cloud_top[i] and close[i] > cloud_bottom[i]) or tk_cross_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters cloud OR TK cross turns bullish
            if (close[i] < cloud_top[i] and close[i] > cloud_bottom[i]) or tk_cross_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_WeeklyFilter"
timeframe = "6h"
leverage = 1.0