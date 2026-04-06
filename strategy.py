#!/usr/bin/env python3
"""
6H Ichimoku Cloud Trend with 1D/1W Filter
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK cross in direction of cloud
captures strong trends. Using 1D for cloud and 1W for higher timeframe filter reduces whipsaw
in both bull and bear markets. Designed for 50-150 trades over 4 years with proper risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1D data for Ichimoku cloud (once before loop)
    df_1d = get_htf_data(prices, '1d')
    # Load 1W data for higher timeframe trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Ichimoku components on 1D
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = np.full(n, np.nan)
    min_low_tenkan = np.full(n, np.nan)
    for i in range(period_tenkan, n):
        max_high_tenkan[i] = np.max(high_1d[i-period_tenkan:i])
        min_low_tenkan[i] = np.min(low_1d[i-period_tenkan:i])
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = np.full(n, np.nan)
    min_low_kijun = np.full(n, np.nan)
    for i in range(period_kijun, n):
        max_high_kijun[i] = np.max(high_1d[i-period_kijun:i])
        min_low_kijun[i] = np.min(low_1d[i-period_kijun:i])
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou = np.full(n, np.nan)
    min_low_senkou = np.full(n, np.nan)
    for i in range(period_senkou_b, n):
        max_high_senkou[i] = np.max(high_1d[i-period_senkou_b:i])
        min_low_senkou[i] = np.min(low_1d[i-period_senkou_b:i])
    senkou_b = (max_high_senkou + min_low_senkou) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # 1W EMA200 for higher timeframe trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = close_1w[i] * 0.01 + ema_200_1w[i-1] * 0.99
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 6H price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of all indicators)
    start = max(52 + 26, 20)  # Senkou B needs 52, plus 26 shift, volume MA 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_6h[i], senkou_b_6h[i])
        cloud_bottom = min(senkou_a_6h[i], senkou_b_6h[i])
        
        # TK Cross
        tk_cross = tenkan_6h[i] - kijun_6h[i]
        tk_cross_prev = tenkan_6h[i-1] - kijun_6h[i-1]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Higher timeframe trend filter (1W EMA200)
        long_trend = close[i] > ema_200_1w_aligned[i]
        short_trend = close[i] < ema_200_1w_aligned[i]
        
        # Volume filter
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: price falls below cloud OR TK cross turns negative
            if close[i] < cloud_bottom or tk_cross < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above cloud OR TK cross turns positive
            if close[i] > cloud_top or tk_cross > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross in direction of cloud + volume + higher timeframe trend
            bull_setup = (tk_cross > 0 and tk_cross_prev <= 0 and  # bullish TK cross
                         price_above_cloud and long_trend and volume_filter)
            bear_setup = (tk_cross < 0 and tk_cross_prev >= 0 and  # bearish TK cross
                         price_below_cloud and short_trend and volume_filter)
            
            if bull_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals