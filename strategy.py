#!/usr/bin/env python3
"""
6h Ichimoku Cloud + Weekly Pivot Direction + Volume Spike
Hypothesis: Combines Ichimoku cloud for trend direction (from 1d), weekly pivot for support/resistance,
and volume spikes to confirm breakouts. Works in bull (long above cloud, bullish pivot) and bear
(short below cloud, bearish pivot). Designed for low trade frequency (~20-40/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku1w_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 1d Ichimoku components (tenkan, kijun, senkou span A/B)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    n_1d = len(close_1d)
    
    # Tenkan-sen (9-period)
    tenkan = np.full(n_1d, np.nan)
    if n_1d >= 9:
        for i in range(8, n_1d):
            tenkan[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (26-period)
    kijun = np.full(n_1d, np.nan)
    if n_1d >= 26:
        for i in range(25, n_1d):
            kijun[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou span A (leading span A)
    senkou_a = np.full(n_1d, np.nan)
    if n_1d >= 26:
        for i in range(25, n_1d):
            senkou_a[i] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou span B (leading span B, 52-period)
    senkou_b = np.full(n_1d, np.nan)
    if n_1d >= 52:
        for i in range(51, n_1d):
            senkou_b[i] = (np.max(high_1d[i-51:i+1]) + np.min(low_1d[i-51:i+1])) / 2
    
    # Cloud top and bottom
    cloud_top = np.full(n_1d, np.nan)
    cloud_bottom = np.full(n_1d, np.nan)
    if n_1d >= 52:
        cloud_top = np.maximum(senkou_a, senkou_b)
        cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Ichimoku trend: price above cloud = bullish, below = bearish
    ichimoku_trend = np.zeros(n_1d)
    if n_1d >= 52:
        ichimoku_trend = np.where(close_1d > cloud_top, 1, np.where(close_1d < cloud_bottom, -1, 0))
    
    # Align Ichimoku trend to 6h
    ichimoku_trend_aligned = align_htf_to_ltf(prices, df_1d, ichimoku_trend)
    
    # Weekly pivot points (from 1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    n_1w = len(close_1w)
    
    # Weekly pivot points (PP, R1, S1, R2, S2)
    pivot = np.full(n_1w, np.nan)
    r1 = np.full(n_1w, np.nan)
    s1 = np.full(n_1w, np.nan)
    r2 = np.full(n_1w, np.nan)
    s2 = np.full(n_1w, np.nan)
    if n_1w >= 1:
        for i in range(n_1w):
            if not (np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i])):
                pivot[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
                r1[i] = 2 * pivot[i] - low_1w[i]
                s1[i] = 2 * pivot[i] - high_1w[i]
                r2[i] = pivot[i] + (high_1w[i] - low_1w[i])
                s2[i] = pivot[i] - (high_1w[i] - low_1w[i])
    
    # Align weekly pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume filter (20-period average)
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need 52 for Ichimoku)
    start = 52
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(ichimoku_trend_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current volume > 2x average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below cloud OR price below S1 pivot OR stoploss
            if (ichimoku_trend_aligned[i] != 1 or
                close[i] < s1_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price above cloud OR price above R1 pivot OR stoploss
            if (ichimoku_trend_aligned[i] != -1 or
                close[i] > r1_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Ichimoku trend + pivot rejection + volume spike
            # Minimum holding period: only allow new entry after 20 bars flat
            if bars_since_entry >= 20:
                # Long: price above cloud, above pivot, rejecting S1 support
                long_condition = (ichimoku_trend_aligned[i] == 1 and
                                close[i] > pivot_aligned[i] and
                                close[i] > s1_aligned[i] and
                                volume_filter)
                
                # Short: price below cloud, below pivot, rejecting R1 resistance
                short_condition = (ichimoku_trend_aligned[i] == -1 and
                                 close[i] < pivot_aligned[i] and
                                 close[i] < r1_aligned[i] and
                                 volume_filter)
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals