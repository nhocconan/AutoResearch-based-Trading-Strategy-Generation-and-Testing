#!/usr/bin/env python3
# 4h_Ichimoku_Cloud_1dTrend_Filter
# Hypothesis: Ichimoku cloud from 1d timeframe provides robust support/resistance and trend direction.
# Tenkan-sen/Kijun-sun cross on 4h triggers entry only when price is above/below 1d cloud.
# Works in bull markets via breakout above cloud and in bear via breakdown below cloud.
# Low trade frequency expected due to dual timeframe confirmation.
# Target: 15-30 trades/year on 4h timeframe.

name = "4h_Ichimoku_Cloud_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ichimoku_cloud(high, low, close):
    """Calculate Ichimoku Cloud components: Tenkan, Kijun, Senkou A/B"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Ichimoku cloud (trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate Ichimoku components on daily timeframe
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = ichimoku_cloud(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Ichomoku components to 4h timeframe (no look-ahead)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Get 4h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Tenkan/Kijun cross on 4h timeframe
    tenkan_4h = pd.Series(high).rolling(window=9, min_periods=9).max()
    tenkan_4h = (tenkan_4h + pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_4h = pd.Series(high).rolling(window=26, min_periods=26).max()
    kijun_4h = (kijun_4h + pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    tenkan_4h = tenkan_4h.values
    kijun_4h = kijun_4h.values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52) + TK cross (26) + vol EMA (20)
    start_idx = max(52, 26, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or
            np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_4h[i]) or
            np.isnan(kijun_4h[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine cloud boundaries (Senkou A/B)
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        if position == 0:
            # Long: Tenkan > Kijun (bullish cross) AND price above cloud
            if tenkan_4h[i] > kijun_4h[i] and close[i] > upper_cloud and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Tenkan < Kijun (bearish cross) AND price below cloud
            elif tenkan_4h[i] < kijun_4h[i] and close[i] < lower_cloud and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan < Kijun OR price falls below cloud
            if tenkan_4h[i] < kijun_4h[i] or close[i] < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan > Kijun OR price rises above cloud
            if tenkan_4h[i] > kijun_4h[i] or close[i] > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals