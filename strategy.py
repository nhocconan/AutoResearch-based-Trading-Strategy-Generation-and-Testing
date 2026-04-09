#!/usr/bin/env python3
# 6h_1d_ichimoku_trend_v1
# Hypothesis: 6-hour Ichimoku trend following with daily cloud filter and volume confirmation.
# Long when price > Kumo (cloud), Tenkan > Kijun, and volume > 1.5x average.
# Short when price < Kumo, Tenkan < Kijun, and volume > 1.5x average.
# Exit when price crosses back into Kumo or Tenkan/Kijun cross reverses.
# Uses daily Ichimoku for major trend, 6h for entry timing. Works in bull/bear via cloud filter.
# Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    tenkan = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            tenkan[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    kijun = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1:
            kijun[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    senkou_b = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= period_senkou_b - 1:
            senkou_b[i] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_a_6h, senkou_b_6h)
    kumo_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price enters Kumo OR Tenkan/Kijun cross reverses (Tenkan < Kijun)
            if (low[i] <= kumo_top[i] and high[i] >= kumo_bottom[i]) or tenkan_6h[i] < kijun_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price enters Kumo OR Tenkan/Kijun cross reverses (Tenkan > Kijun)
            if (low[i] <= kumo_top[i] and high[i] >= kumo_bottom[i]) or tenkan_6h[i] > kijun_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price > Kumo, Tenkan > Kijun, volume confirmation
            if (low[i] > kumo_top[i] and 
                tenkan_6h[i] > kijun_6h[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price < Kumo, Tenkan < Kijun, volume confirmation
            elif (high[i] < kumo_bottom[i] and 
                  tenkan_6h[i] < kijun_6h[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals