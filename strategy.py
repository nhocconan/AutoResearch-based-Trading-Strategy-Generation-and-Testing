#!/usr/bin/env python3
# 6H_Ichimoku_TK_Cross_CloudFilter_1DTrend_Volume
# Hypothesis: Combines Ichimoku TK Cross (Tenkan/Kijun) with 1-day cloud filter and volume confirmation.
# Uses 1-day Ichimoku cloud to establish trend direction (price above/below cloud) and 6h TK cross for entry timing.
# Volume filter ensures momentum confirmation. Designed for 6h timeframe with low trade frequency (<30/year).
# Works in both bull/bear markets: cloud filter avoids counter-trend trades, TK cross captures momentum shifts.

name = "6H_Ichimoku_TK_Cross_CloudFilter_1DTrend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need enough data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_1d).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_1d).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_1d).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_1d).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_1d).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for cloud)
    
    # Determine cloud (Kumo) boundaries: upper = max(Senkou A, B), lower = min(Senkou A, B)
    # For trend filter: price > cloud top = uptrend, price < cloud bottom = downtrend
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    cloud_top_aligned = align_htf_to_ltf(prices, df_1d, cloud_top)
    cloud_bottom_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom)
    
    # TK Cross signals on 1d (for entry timing)
    # Bullish cross: Tenkan crosses above Kijun
    # Bearish cross: Tenkan crosses below Kijun
    tk_cross_bull = np.zeros(len(tenkan), dtype=bool)
    tk_cross_bear = np.zeros(len(tenkan), dtype=bool)
    for i in range(1, len(tenkan)):
        if not (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(tenkan[i-1]) or np.isnan(kijun[i-1])):
            tk_cross_bull[i] = (tenkan[i] > kijun[i]) and (tenkan[i-1] <= kijun[i-1])
            tk_cross_bear[i] = (tenkan[i] < kijun[i]) and (tenkan[i-1] >= kijun[i-1])
    
    tk_cross_bull_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_bull.astype(float))
    tk_cross_bear_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_bear.astype(float))
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 26)  # Ensure we have volume MA and Ichimoku data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(tk_cross_bull_aligned[i]) or np.isnan(tk_cross_bear_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bullish TK cross + price above cloud (uptrend) + volume spike
            if (tk_cross_bull_aligned[i] > 0.5 and  # True signal
                close[i] > cloud_top_aligned[i] and   # Above cloud = uptrend
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bearish TK cross + price below cloud (downtrend) + volume spike
            elif (tk_cross_bear_aligned[i] > 0.5 and  # True signal
                  close[i] < cloud_bottom_aligned[i] and  # Below cloud = downtrend
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Opposite TK cross (trend reversal signal)
            # 2. Price crosses back into cloud (trend weakness)
            opposite_cross = (position == 1 and tk_cross_bear_aligned[i] > 0.5) or \
                           (position == -1 and tk_cross_bull_aligned[i] > 0.5)
            price_in_cloud = (close[i] <= cloud_top_aligned[i] and close[i] >= cloud_bottom_aligned[i])
            
            if opposite_cross or price_in_cloud:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals