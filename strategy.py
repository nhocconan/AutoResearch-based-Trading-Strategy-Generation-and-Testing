#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_v1
Hypothesis: Use Ichimoku cloud from 1d timeframe as a trend filter and Senkou Span crossover for entry signals. 
Long when Tenkan-sen crosses above Kijun-sen AND price is above the cloud (bullish trend).
Short when Tenkan-sen crosses below Kijun-sen AND price is below the cloud (bearish trend).
Exit when price crosses back through the Kijun-sen or when the cloud twists (Senkou A/B crossover).
Ichimoku provides multi-dimensional trend, support/resistance, and momentum in one system.
Works in bull markets (buy on bullish TK cross above cloud) and bear markets (sell on bearish TK cross below cloud).
Target: 20-40 trades per year on 6h timeframe.
"""

name = "6h_Ichimoku_Cloud_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods behind
    # Not used for signals as it requires future data
    
    # Align 1D Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or 
            np.isnan(senkou_b_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Cloud twist: Senkou A/B crossover (trend change signal)
        # We'll use this for exit conditions
        senkou_a_prev = senkou_a_aligned[i-1] if i > 0 else senkou_a_aligned[i]
        senkou_b_prev = senkou_b_aligned[i-1] if i > 0 else senkou_b_aligned[i]
        cloud_twist_bullish = senkou_a_aligned[i] > senkou_b_aligned[i] and senkou_a_prev <= senkou_b_prev
        cloud_twist_bearish = senkou_a_aligned[i] < senkou_b_aligned[i] and senkou_a_prev >= senkou_b_prev
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud (bullish)
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_prev <= kijun_prev
            
            if tk_cross_bullish and close[i] > cloud_top:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud (bearish)
            elif tenkan_aligned[i] < kijun_aligned[i] and tenkan_prev >= kijun_prev:
                if close[i] < cloud_bottom:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud bottom OR cloud turns bearish
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            tk_cross_bearish = tenkan_aligned[i] < kijun_aligned[i] and tenkan_prev >= kijun_prev
            
            if tk_cross_bearish or close[i] < cloud_bottom or cloud_twist_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud top OR cloud turns bullish
            tenkan_prev = tenkan_aligned[i-1] if i > 0 else tenkan_aligned[i]
            kijun_prev = kijun_aligned[i-1] if i > 0 else kijun_aligned[i]
            tk_cross_bullish = tenkan_aligned[i] > kijun_aligned[i] and tenkan_prev <= kijun_prev
            
            if tk_cross_bullish or close[i] > cloud_top or cloud_twist_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals