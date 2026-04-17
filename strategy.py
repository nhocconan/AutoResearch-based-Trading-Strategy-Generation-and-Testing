#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Ichimoku Cloud on 1d timeframe as trend filter with TK cross entry on 6h.
- Cloud color (green/red) determines trend direction from daily Ichimoku
- TK cross (Tenkan/Kijun) on 6h provides entry timing in direction of daily trend
- Requires price to be outside cloud to avoid whipsaws in consolidation
- Designed to work in both bull and bear markets by following higher timeframe trend
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
    n = len(high)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.maximum.accumulate(high)
    period9_low = np.minimum.accumulate(low)
    # For proper windowing, we need to use rolling max/min
    tenkan = np.full(n, np.nan)
    kijun = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 8:  # 9 periods
            tenkan[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
        if i >= 26:  # 26 periods
            kijun[i] = (np.max(high[i-26:i+1]) + np.min(low[i-26:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            idx = i + 26
            if idx < n:
                senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = np.full(n, np.nan)
    for i in range(n):
        if i >= 51:  # 52 periods
            senkou_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
            idx = i + 26
            if idx < n:
                senkou_b[idx] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Chikou Span (Lagging Span): Close shifted 26 periods back
    chikou = np.full(n, np.nan)
    for i in range(26, n):
        chikou[i-26] = close[i]
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Ichimoku for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Determine cloud color and trend
    # Green cloud (bullish): Senkou A > Senkou B
    # Red cloud (bearish): Senkou A < Senkou B
    cloud_bullish = senkou_a_1d > senkou_b_1d
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    cloud_bullish_aligned = align_htf_to_ltf(prices, df_1d, cloud_bullish.astype(float))
    
    # === 6h TK Cross for entry timing ===
    # Tenkan-sen (9-period) on 6h
    tenkan_6h = np.full(n, np.nan)
    kijun_6h = np.full(n, np.nan)
    for i in range(n):
        if i >= 8:  # 9 periods
            tenkan_6h[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
        if i >= 26:  # 26 periods
            kijun_6h[i] = (np.max(high[i-26:i+1]) + np.min(low[i-26:i+1])) / 2
    
    tk_cross = tenkan_6h - kijun_6h  # Positive when Tenkan > Kijun (bullish cross)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 20:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or 
            np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(cloud_bullish_aligned[i]) or
            np.isnan(tenkan_6h[i]) or 
            np.isnan(kijun_6h[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine cloud color and price position relative to cloud
        is_green_cloud = cloud_bullish_aligned[i] > 0.5
        senkou_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        senkou_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > senkou_top
        price_below_cloud = close[i] < senkou_bottom
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Green cloud on 1d AND price above cloud AND TK cross bullish (Tenkan > Kijun) AND volume
            if (is_green_cloud and 
                price_above_cloud and 
                tk_cross[i] > 0 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Red cloud on 1d AND price below cloud AND TK cross bearish (Tenkan < Kijun) AND volume
            elif (not is_green_cloud and 
                  price_below_cloud and 
                  tk_cross[i] < 0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Red cloud OR price drops below cloud OR TK cross turns bearish
            if (not is_green_cloud or 
                not price_above_cloud or 
                tk_cross[i] < 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Green cloud OR price rises above cloud OR TK cross turns bullish
            if (is_green_cloud or 
                not price_below_cloud or 
                tk_cross[i] > 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0