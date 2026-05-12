#!/usr/bin/env python3
# 6h_IchimokuCloud_1dTrend_Volume
# Hypothesis: Use Ichimoku Cloud from 1d to determine trend, Tenkan/Kijun cross from 1d for entry signals, and volume confirmation on 6h.
# Enter long when Tenkan crosses above Kijun and price is above cloud, short when Tenkan crosses below Kijun and price is below cloud.
# Only trade in direction of 1d Ichimoku trend (price above/below cloud).
# Exit when price crosses back through Kijun line or cloud.
# Designed for low frequency (12-30 trades/year) by using 1d for signal generation and 6h only for execution timing.
# Works in both bull and bear markets by following higher timeframe Ichimoku trend.

name = "6h_IchimokuCloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1d data for Ichimoku calculations ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_1d = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_1d = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low) / 2
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b_1d = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h (wait for 1d bar to close)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume confirmation (20-period average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ensure Ichimoku is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Trend filter: price above/below cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Tenkan/Kijun cross signals
        tenkan_prev = tenkan_1d_aligned[i-1]
        kijun_prev = kijun_1d_aligned[i-1]
        tenkan_curr = tenkan_1d_aligned[i]
        kijun_curr = kijun_1d_aligned[i]
        
        tk_cross_up = tenkan_prev <= kijun_prev and tenkan_curr > kijun_curr
        tk_cross_down = tenkan_prev >= kijun_prev and tenkan_curr < kijun_curr
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Tenkan crosses above Kijun and price above cloud
            if tk_cross_up and price_above_cloud and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun and price below cloud
            elif tk_cross_down and price_below_cloud and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price crosses below Kijun or enters cloud
            if close[i] < kijun_1d_aligned[i] or close[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Kijun or enters cloud
            if close[i] > kijun_1d_aligned[i] or close[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals