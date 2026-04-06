#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d Trend Filter + Volume Confirmation
Hypothesis: Ichimoku TK cross provides timely signals, cloud from 1d filters trend direction, volume confirms momentum. Works in bull/bear by only trading in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 10, 26, 52 periods for Ichimoku
    tenkan_sen = np.full(n, np.nan)
    kijun_sen = np.full(n, np.nan)
    senkou_span_a = np.full(n, np.nan)
    senkou_span_b = np.full(n, np.nan)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    if n >= 9:
        for i in range(8, n):
            tenkan_sen[i] = (np.max(high[i-8:i+1]) + np.min(low[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    if n >= 26:
        for i in range(25, n):
            kijun_sen[i] = (np.max(high[i-25:i+1]) + np.min(low[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    if n >= 26:
        for i in range(26, n):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    if n >= 52:
        for i in range(51, n):
            senkou_span_b[i] = (np.max(high[i-51:i+1]) + np.min(low[i-51:i+1])) / 2
    
    # Load 1d trend: EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: 20-period average
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from 52 to ensure all Ichimoku components available
    start = 52
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top and bottom (shifted 26 periods ahead)
        # Since we don't have future data, we use current Senkou spans as cloud
        # In real Ichimoku, cloud is plotted 26 periods ahead, but for filtering we use current values
        cloud_top = max(senkou_span_a[i], senkou_span_b[i])
        cloud_bottom = min(senkou_span_a[i], senkou_span_b[i])
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        if position == 1:  # long position
            # Exit: TK cross down OR price falls below cloud bottom
            tk_cross_down = tenkan_sen[i] < kijun_sen[i]
            below_cloud = close[i] < cloud_bottom
            
            if tk_cross_down or below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross up OR price rises above cloud top
            tk_cross_up = tenkan_sen[i] > kijun_sen[i]
            above_cloud = close[i] > cloud_top
            
            if tk_cross_up or above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price relative to cloud + 1d trend + volume
            tk_cross_up = tenkan_sen[i] > kijun_sen[i]
            tk_cross_down = tenkan_sen[i] < kijun_sen[i]
            
            price_above_cloud = close[i] > cloud_top
            price_below_cloud = close[i] < cloud_bottom
            
            # 1d trend filter
            trend_up = close[i] > ema_1d_aligned[i]
            trend_down = close[i] < ema_1d_aligned[i]
            
            if tk_cross_up and price_above_cloud and trend_up and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif tk_cross_down and price_below_cloud and trend_down and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals