#!/usr/bin/env python3
"""
1d_ICHIMOKU_TENKAN_KIJUN_CROSS_WEEKLYTREND_VOLUME
Hypothesis: Use Ichimoku Tenkan/Kijun cross on 1d with 1w trend filter (price above/below Kumo cloud) and volume confirmation. Works in bull markets (buy when Tenkan > Kijun in bullish trend) and bear markets (sell when Tenkan < Kijun in bearish trend). Volume confirms breakout strength. Target: 10-25 trades per year on 1d timeframe.
"""

name = "1d_ICHIMOKU_TENKAN_KIJUN_CROSS_WEEKLYTREND_VOLUME"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1W Data for Trend Filter (Kumo Cloud) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Calculate Tenkan-sen
    high_tenkan = pd.Series(high_1w).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    low_tenkan = pd.Series(low_1w).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Calculate Kijun-sen (Base Line)
    high_kijun = pd.Series(high_1w).rolling(window=kijun_period, min_periods=kijun_period).max().values
    low_kijun = pd.Series(low_1w).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Calculate Senkou Span B (Leading Span B)
    high_senkou_b = pd.Series(high_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    low_senkou_b = pd.Series(low_1w).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # Align Ichimoku components to 1d timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # Determine Kumo cloud boundaries (Senkou Span A/B)
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Trend filter: price above/below Kumo cloud
    price_above_kumo = close > kumo_top
    price_below_kumo = close < kumo_bottom
    
    # === 1D Data for Ichimoku Cross Signal ===
    # Tenkan-sen on daily data
    high_tenkan_d = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    low_tenkan_d = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen_d = (high_tenkan_d + low_tenkan_d) / 2
    
    # Kijun-sen on daily data
    high_kijun_d = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    low_kijun_d = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen_d = (high_kijun_d + low_kijun_d) / 2
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(tenkan_period, kijun_period, senkou_span_b_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(kumo_top[i]) or 
            np.isnan(kumo_bottom[i]) or
            np.isnan(tenkan_sen_d[i]) or 
            np.isnan(kijun_sen_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above Kumo cloud AND volume spike
            if (tenkan_sen_d[i] > kijun_sen_d[i] and 
                tenkan_sen_d[i-1] <= kijun_sen_d[i-1] and  # crossover confirmation
                price_above_kumo[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below Kumo cloud AND volume spike
            elif (tenkan_sen_d[i] < kijun_sen_d[i] and 
                  tenkan_sen_d[i-1] >= kijun_sen_d[i-1] and  # crossover confirmation
                  price_below_kumo[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price falls below Kumo cloud
            if (tenkan_sen_d[i] < kijun_sen_d[i] and 
                tenkan_sen_d[i-1] >= kijun_sen_d[i-1]) or not price_above_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above Kumo cloud
            if (tenkan_sen_d[i] > kijun_sen_d[i] and 
                tenkan_sen_d[i-1] <= kijun_sen_d[i-1]) or not price_below_kumo[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals