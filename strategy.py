#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_Filter_v2
Hypothesis: 6h Ichimoku Kumo twist (Tenkan/Kijun cross) with 1d trend filter (price vs Kumo) and volume confirmation.
Long when Tenkan crosses above Kijun, price is above Kumo (bullish), and volume > 1.5x 20-period average.
Short when Tenkan crosses below Kijun, price is below Kumo (bearish), and volume > 1.5x 20-period average.
Exit on opposite Kumo twist or trend reversal (price crosses Kumo).
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year on 6h.
Works in bull via trend-following Kumo twists, in bear via mean reversion at Kumo edges.
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
    
    # Get 6h data for Ichimoku calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # need 52 for Senkou Span B
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components to original timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_b)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Get 1d data for trend filter (price vs Kumo from 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Ichimoku Kumo for trend filter
    high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_tenkan_1d + low_tenkan_1d) / 2
    
    high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_kijun_1d + low_kijun_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_senkou_b_1d + low_senkou_b_1d) / 2
    
    # Align 1d Kumo to original timeframe
    kumo_top_1d = align_htf_to_ltf(prices, df_1d, np.maximum(senkou_a_1d, senkou_b_1d))
    kumo_bottom_1d = align_htf_to_ltf(prices, df_1d, np.minimum(senkou_a_1d, senkou_b_1d))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or
            np.isnan(kumo_top_1d[i]) or np.isnan(kumo_bottom_1d[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Kumo twist signals
        tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
        kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
        tenkan_cross_above = tenkan_prev <= kijun_prev and tenkan_aligned[i] > kijun_aligned[i]
        tenkan_cross_below = tenkan_prev >= kijun_prev and tenkan_aligned[i] < kijun_aligned[i]
        
        # Trend filter: price vs 1d Kumo
        price_above_1d_kumo = close[i] > kumo_top_1d[i]
        price_below_1d_kumo = close[i] < kumo_bottom_1d[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun, price above 1d Kumo, volume spike
            long_signal = tenkan_cross_above and price_above_1d_kumo and vol_spike[i]
            # Short: Tenkan crosses below Kijun, price below 1d Kumo, volume spike
            short_signal = tenkan_cross_below and price_below_1d_kumo and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: Tenkan crosses below Kijun OR price crosses below 1d Kumo
            exit_signal = tenkan_cross_below or (close[i] < kumo_bottom_1d[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: Tenkan crosses above Kijun OR price crosses above 1d Kumo
            exit_signal = tenkan_cross_above or (close[i] > kumo_top_1d[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_Filter_v2"
timeframe = "6h"
leverage = 1.0