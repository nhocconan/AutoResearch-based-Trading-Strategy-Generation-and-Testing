#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Regime_1dTrend_v1
Hypothesis: Use 1d Ichimoku cloud as trend filter (price above/below cloud) and TK cross on 6h for entry timing. Only trade in direction of 1d cloud to avoid counter-trend whipsaws. Volume confirmation (1.5x median) ensures momentum. Target: 20-40 trades/year on 6h. Works in bull/bear by following higher timeframe trend.
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
    
    # Get 1d data for HTF Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 26*2 for Ichimoku
        return np.zeros(n)
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align HTF Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # 6h TK cross (Tenkan-sen crosses Kijun-sen)
    # Calculate 6h Tenkan-sen and Kijun-sen for entry timing
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Volume confirmation: 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of Ichimoku calculations (52) and volume median (20)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_sen_6h[i]) or
            np.isnan(kijun_sen_6h[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d trend: price relative to cloud
        # Cloud top is max(Senkou Span A, Senkou Span B)
        # Cloud bottom is min(Senkou Span A, Senkou Span B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        price = close[i]
        
        # Price above cloud = bullish trend, below cloud = bearish trend
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom
        
        # 6h TK cross signals
        tk_cross_up = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
        tk_cross_down = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
        
        vol_signal = volume[i] > 1.5 * vol_median[i]
        
        if position == 0:
            # Long: price above cloud (bullish 1d trend) + TK cross up + volume
            if above_cloud and tk_cross_up and vol_signal:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud (bearish 1d trend) + TK cross down + volume
            elif below_cloud and tk_cross_down and vol_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long until TK cross down or price breaks below cloud
            signals[i] = 0.25
            if tk_cross_down or price < cloud_bottom:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short until TK cross up or price breaks above cloud
            signals[i] = -0.25
            if tk_cross_up or price > cloud_top:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Regime_1dTrend_v1"
timeframe = "6h"
leverage = 1.0