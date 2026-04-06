#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 1d Ichimoku cloud filter + volume confirmation
Hypothesis: Ichimoku cloud on daily timeframe provides strong trend filter for 6h breakouts.
Cloud acts as dynamic support/resistance - price above cloud = bullish bias, below = bearish.
Volume confirmation ensures breakouts have conviction. Works in bull (continuation) and bear (reversals at cloud edges).
Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_vol_v1"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan = 9
    kijun = 26
    senkou = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    def calculate_tenkan(high_arr, low_arr, period):
        tenkan_vals = np.full_like(high_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            tenkan_vals[i] = (np.max(high_1d[i-period+1:i+1]) + np.min(low_1d[i-period+1:i+1])) / 2
        return tenkan_vals
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    def calculate_kijun(high_arr, low_arr, period):
        kijun_vals = np.full_like(high_arr, np.nan)
        for i in range(period-1, len(high_arr)):
            kijun_vals[i] = (np.max(high_1d[i-period+1:i+1]) + np.min(low_1d[i-period+1:i+1])) / 2
        return kijun_vals
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    
    tenkan_1d = calculate_tenkan(high_1d, low_1d, tenkan)
    kijun_1d = calculate_kijun(high_1d, low_1d, kijun)
    
    # Senkou Span A
    senkou_a = np.full_like(high_1d, np.nan)
    for i in range(len(tenkan_1d)):
        if not np.isnan(tenkan_1d[i]) and not np.isnan(kijun_1d[i]):
            senkou_a[i] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    # Senkou Span B
    senkou_b = np.full_like(high_1d, np.nan)
    for i in range(senkou-1, len(high_1d)):
        senkou_b[i] = (np.max(high_1d[i-senkou+1:i+1]) + np.min(low_1d[i-senkou+1:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe (with shift for look-ahead prevention)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Donchian channels (20-period) on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 26, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss hit
            if (close[i] < donchian_low[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss hit
            if (close[i] > donchian_high[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high, above cloud, with volume
            if (close[i] > donchian_high[i] and 
                close[i] > cloud_top and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, below cloud, with volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < cloud_bottom and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals