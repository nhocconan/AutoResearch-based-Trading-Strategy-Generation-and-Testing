#!/usr/bin/env python3
"""
6h_Ichimoku_KijunTenkan_Cross_CloudFilter_1dTrend_Volume
Hypothesis: On 6h timeframe, Ichimoku Tenkan/Kijun cross provides entry signals,
filtered by 1d cloud (Senkou Span A/B) for trend direction and volume confirmation.
Works in bull markets when price is above cloud (uptrend) and bear markets when
price is below cloud (downtrend). Cloud acts as dynamic support/resistance.
Tenkan/Kijun cross captures momentum shifts. Volume filters false signals.
Targets 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
"""

name = "6h_Ichimoku_KijunTenkan_Cross_CloudFilter_1dTrend_Volume"
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
    volume = prices['volume'].values
    
    # === 1d Data for Ichimoku Components ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    max_high_52 = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_6h, senkou_b_6h)
    cloud_bottom = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # === Volume Filter (1.5x 20-period EMA on 6h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Ichimoku calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Tenkan crosses above Kijun AND price above cloud (uptrend) AND volume
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1] and
                close[i] > cloud_top[i] and volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: Tenkan crosses below Kijun AND price below cloud (downtrend) AND volume
            elif (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1] and
                  close[i] < cloud_bottom[i] and volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price drops below cloud bottom
            if (tenkan_6h[i] < kijun_6h[i] and tenkan_6h[i-1] >= kijun_6h[i-1]) or \
               close[i] < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price rises above cloud top
            if (tenkan_6h[i] > kijun_6h[i] and tenkan_6h[i-1] <= kijun_6h[i-1]) or \
               close[i] > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals