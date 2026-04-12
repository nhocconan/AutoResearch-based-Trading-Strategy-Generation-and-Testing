#!/usr/bin/env python3
"""
6h_1w_Ichimoku_Trend_v1
Hypothesis: Ichimoku cloud with daily conversion/base line cross and weekly trend filter.
Long when price above cloud (bullish), Tenkan > Kijun, and weekly close above weekly Kumo top.
Short when price below cloud (bearish), Tenkan < Kijun, and weekly close below weekly Kumo bottom.
Uses Ichimoku's multi-line structure to filter false signals in ranging markets.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
Works in bull (follow cloud direction) and bear (counter-trend at cloud edges).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Ichimoku_Trend_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span"""
    n = len(close)
    if n < senkou:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max().values
    tenkan_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min().values
    tenkan = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max().values
    kijun_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min().values
    kijun = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max().values
    senkou_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min().values
    senkou_b = ((senkou_high + senkou_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = np.roll(close, -kijun)  # Negative shift for lagging
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Ichimoku on 6h data
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close, 9, 26, 52)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Weekly Ichimoku for trend filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    wk_tenkan, wk_kijun, wk_senkou_a, wk_senkou_b, _ = calculate_ichimoku(weekly_high, weekly_low, weekly_close, 9, 26, 52)
    
    # Align weekly Ichimoku to 6h
    wk_tenkan_aligned = align_htf_to_ltf(prices, df_1w, wk_tenkan)
    wk_kijun_aligned = align_htf_to_ltf(prices, df_1w, wk_kijun)
    wk_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_a)
    wk_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any data invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(wk_tenkan_aligned[i]) or 
            np.isnan(wk_kijun_aligned[i]) or np.isnan(wk_senkou_a_aligned[i]) or 
            np.isnan(wk_senkou_b_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine cloud boundaries (Senkou Span A/B)
        cloud_top = np.maximum(senkou_a[i], senkou_b[i])
        cloud_bottom = np.minimum(senkou_a[i], senkou_b[i])
        
        # Weekly cloud boundaries
        wk_cloud_top = np.maximum(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
        wk_cloud_bottom = np.minimum(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
        
        # Ichimoku signals
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Weekly trend filter: price relative to weekly cloud
        weekly_close_price = df_1w['close'].values
        wk_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_price)
        weekly_bullish = wk_close_aligned[i] > wk_cloud_top
        weekly_bearish = wk_close_aligned[i] < wk_cloud_bottom
        
        # Entry logic
        long_entry = price_above_cloud and tenkan_above_kijun and weekly_bullish
        short_entry = price_below_cloud and tenkan_below_kijun and weekly_bearish
        
        # Exit logic: opposite signal or cloud penetration
        long_exit = price_below_cloud or tenkan_below_kijun
        short_exit = price_above_cloud or tenkan_above_kijun
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals