#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
- Uses Ichimoku Cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h timeframe
- Long signal: Tenkan crosses above Kijun + price above Cloud + volume > 1.5x 20-period avg + 1d EMA50 uptrend
- Short signal: Tenkan crosses below Kijun + price below Cloud + volume > 1.5x 20-period avg + 1d EMA50 downtrend
- Exit: Tenkan-Kijun cross in opposite direction or price re-enters Cloud
- Ichimoku provides dynamic support/resistance and trend identification that works across market regimes
- Volume confirmation reduces false signals
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
"""

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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2.0
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (need 52 for Senkou B)
    start_idx = max(52, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(tenkan[i]) or 
            np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine Cloud boundaries (Senkou Span A/B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Tenkan-Kijun cross signals
        tenkan_prev = tenkan[i-1] if i > 0 else tenkan[i]
        kijun_prev = kijun[i-1] if i > 0 else kijun[i]
        tenkan_cross_above = tenkan[i] > kijun[i] and tenkan_prev <= kijun_prev
        tenkan_cross_below = tenkan[i] < kijun[i] and tenkan_prev >= kijun_prev
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun + price above Cloud + volume spike + 1d uptrend
            if (tenkan_cross_above and 
                close[i] > upper_cloud and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun + price below Cloud + volume spike + 1d downtrend
            elif (tenkan_cross_below and 
                  close[i] < lower_cloud and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun OR price re-enters Cloud
            if tenkan_cross_below or close[i] < upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun OR price re-enters Cloud
            if tenkan_cross_above or close[i] > lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0