#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeConfirm
Hypothesis: Ichimoku TK cross with 1d cloud filter and volume confirmation on 6h timeframe.
Uses TK cross (Tenkan/Kijun) for momentum signals, filtered by price position relative to 1d Ichimoku cloud (trend direction).
Volume spike confirms institutional participation. Designed for low turnover (12-37 trades/year) with discrete sizing (0.25) to minimize fee drag.
Works in bull/bear markets by following 1d trend via cloud filter while using 6h TK cross for timely entries.
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
    
    # Get 1d data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
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
    
    # Align 1d Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # 6h Ichimoku for TK cross (signal line)
    period_tenkan_6h = 9
    period_kijun_6h = 26
    max_high_tenkan_6h = pd.Series(high).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).max().values
    min_low_tenkan_6h = pd.Series(low).rolling(window=period_tenkan_6h, min_periods=period_tenkan_6h).min().values
    tenkan_6h = (max_high_tenkan_6h + min_low_tenkan_6h) / 2
    
    max_high_kijun_6h = pd.Series(high).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).max().values
    min_low_kijun_6h = pd.Series(low).rolling(window=period_kijun_6h, min_periods=period_kijun_6h).min().values
    kijun_6h = (max_high_kijun_6h + min_low_kijun_6h) / 2
    
    # TK cross signals: Tenkan crosses above/below Kijun
    tk_cross_above = (tenkan_6h > kijun_6h) & (np.roll(tenkan_6h, 1) <= np.roll(kijun_6h, 1))
    tk_cross_below = (tenkan_6h < kijun_6h) & (np.roll(tenkan_6h, 1) >= np.roll(kijun_6h, 1))
    
    # Handle first element for roll
    tk_cross_above[0] = False
    tk_cross_below[0] = False
    
    # Cloud trend filter: price above/below cloud
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d Ichimoku (52) + 6h TK (26) + volume avg (20)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        price = close[i]
        tenkan_t = tenkan_6h[i]
        kijun_t = kijun_6h[i]
        tenkan_y = np.roll(tenkan_6h, 1)[i]
        kijun_y = np.roll(kijun_6h, 1)[i]
        price_above = price_above_cloud[i]
        price_below = price_below_cloud[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: TK bullish cross + price above 1d cloud + volume confirmation
            long_condition = tk_cross_above[i] and price_above and vol_conf
            # Short: TK bearish cross + price below 1d cloud + volume confirmation
            short_condition = tk_cross_below[i] and price_below and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: TK bearish cross OR price breaks below cloud bottom
            exit_condition = tk_cross_below[i] or (price < cloud_bottom[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TK bullish cross OR price breaks above cloud top
            exit_condition = tk_cross_above[i] or (price > cloud_top[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0