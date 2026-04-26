#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: Ichimoku cloud (from 1d) acts as dynamic support/resistance on 6h. Price breaking above/below cloud with volume spike and aligned with 1d trend (via Tenkan/Kijun cross) captures strong continuation moves. Cloud filter reduces whipsaws in ranging markets. Works in bull/bear by only trading in direction of higher timeframe trend. Target: 60-120 trades over 4 years (15-30/year).
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
    
    # Load 1d data ONCE before loop for Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Need sufficient 1d data for Ichimoku (52 periods max)
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not needed for breakout)
    
    # Future cloud: Senkou A/B shifted 26 periods ahead
    # But for current cloud, we use values already shifted
    # Ichimoku cloud = between Senkou A and Senkou B
    # Align all to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # The actual cloud boundaries for current period are Senkou A/B from 26 periods ago
    # So we need to shift the aligned arrays by 26* (1d -> 6h conversion)
    # 1d = 4 * 6h bars, so 26 * 4 = 104 bars of 6h
    # But align_htf_to_ltf already accounts for HTF bar completion, so we use the values as-is
    # The cloud is plotted 26 periods ahead, so current cloud is from past calculation
    # For simplicity, we use current Tenkan/Kijun/Senkou for trend and cloud edges
    
    # Trend filter: Tenkan > Kijun = uptrend, Tenkan < Kijun = downtrend
    # Cloud top/bottom: max/min of Senkou A/B
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume spike detection on 6h (volume > 2.0x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 26, 20)  # 26 for Ichimoku base, 20 for volume EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter via Tenkan/Kijun
        uptrend = tenkan_aligned[i] > kijun_aligned[i]
        downtrend = tenkan_aligned[i] < kijun_aligned[i]
        
        # Cloud breakout logic
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        # Long logic: price breaks above cloud with volume spike + in uptrend
        if price_above_cloud and volume_spike[i] and uptrend:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below cloud with volume spike + in downtrend
        elif price_below_cloud and volume_spike[i] and downtrend:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite cloud side or trend weakens
        elif position == 1 and (price_below_cloud or not uptrend):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (price_above_cloud or not downtrend):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0