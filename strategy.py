#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_CloudFilter_1wTrend_VolumeSpike
Hypothesis: On 6h timeframe, use Ichimoku Tenkan-Kijun cross from 1d as entry trigger, with 1w trend filter (price > 1w Kumo top/bottom) and volume confirmation (>2.0x 24-period average). This combines momentum (TK cross) with higher timeframe structure (1w cloud) and volume validation to capture strong trend continuations while avoiding whipsaws in ranging markets. Target: 12-25 trades/year.
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
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:  # Need at least 26 periods for 1w Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Calculate 1w Ichimoku cloud for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Tenkan-sen (9-period)
    high_tenkan_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_tenkan_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_tenkan_1w + low_tenkan_1w) / 2
    
    # 1w Kijun-sen (26-period)
    high_kijun_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_kijun_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_kijun_1w + low_kijun_1w) / 2
    
    # 1w Senkou Span A
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    # 1w Senkou Span B (52-period)
    high_senkou_b_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_senkou_b_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_senkou_b_1w + low_senkou_b_1w) / 2
    
    # Kumo top/bottom for 1w trend filter
    kumo_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    kumo_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Align 1d Ichimoku to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Align 1w Kumo to 6h timeframe
    kumo_top_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_top_1w)
    kumo_bottom_1w_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom_1w)
    
    # Volume confirmation: volume > 2.0x 24-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52 periods for Senkou B) + volume MA warmup
    start_idx = max(52 + 26, 24)  # 52 for Senkou B + 26 shift, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(kumo_top_1w_aligned[i]) or np.isnan(kumo_bottom_1w_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Kumo thickness filter (avoid thin/cloudless conditions)
        kumo_thickness = np.abs(senkou_a_aligned[i] - senkou_b_aligned[i])
        price_level = close[i]
        kumo_thick_enough = kumo_thickness > (price_level * 0.005)  # At least 0.5% of price
        
        # 1w trend alignment
        price_above_kumo = price_level > kumo_top_1w_aligned[i]
        price_below_kumo = price_level < kumo_bottom_1w_aligned[i]
        
        if position == 0:
            # Long: TK cross bullish (Tenkan > Kijun) + price above 1w Kumo + volume spike + thick cloud
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            long_signal = tk_bullish and price_above_kumo and volume_spike[i] and kumo_thick_enough
            
            # Short: TK cross bearish (Tenkan < Kijun) + price below 1w Kumo + volume spike + thick cloud
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
            short_signal = tk_bearish and price_below_kumo and volume_spike[i] and kumo_thick_enough
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: TK cross bearish OR price drops below 1w Kumo bottom
            tk_bearish = tenkan_aligned[i] < kijun_aligned[i]
            if tk_bearish or not price_above_kumo:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TK cross bullish OR price rises above 1w Kumo top
            tk_bullish = tenkan_aligned[i] > kijun_aligned[i]
            if tk_bullish or not price_below_kumo:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_CloudFilter_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0