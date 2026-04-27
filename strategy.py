#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_VolumeSpike_HTF
Hypothesis: Ichimoku cloud twist (Tenkan/Kijun cross) on 6h with 1w trend filter (price vs Senkou Span B) and volume confirmation.
Works in bull/bear by following 1w trend: long only when price > Senkou Span B (bullish 1w), short only when price < Senkou Span B (bearish 1w).
Cloud twist catches momentum shifts; volume confirms authenticity. Targets 12-30 trades/year on 6h to avoid fee drag.
Uses actual Ichimoku formulas with proper lookback periods.
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
    
    # Ichimoku parameters (standard)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    highest_9 = pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values
    lowest_9 = pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values
    tenkan_sen = (highest_9 + lowest_9) / 2
    
    # Calculate Kijun-sen (Base Line): (26-period high + 26-period low)/2
    highest_26 = pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values
    lowest_26 = pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values
    kijun_sen = (highest_26 + lowest_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 displaced 26 periods ahead
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 displaced 26 periods ahead
    highest_52 = pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_52 = pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b = (highest_52 + lowest_52) / 2
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    # Calculate 1w Senkou Span B for trend filter (price vs Senkou Span B indicates 1w trend)
    highest_52_1w = pd.Series(df_1w['high']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values
    lowest_52_1w = pd.Series(df_1w['low']).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values
    senkou_span_b_1w = (highest_52_1w + lowest_52_1w) / 2
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position size
    
    # Warmup: need enough for Ichimoku calculations (max 52 + 26 displacement)
    start_idx = max(100, senkou_span_b_period + displacement, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(senkou_span_b_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        
        # Ichimoku signals
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        span_a = senkou_span_a[i]
        span_b = senkou_span_b[i]
        span_b_1w = senkou_span_b_1w_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Tenkan/Kijun cross (twist)
        # We need previous values to detect cross
        if i > 0:
            prev_tenkan = tenkan_sen[i-1]
            prev_kijun = kijun_sen[i-1]
            tenkan_cross_above = tenkan > kijun and prev_tenkan <= prev_kijun
            tenkan_cross_below = tenkan < kijun and prev_tenkan >= prev_kijun
        else:
            tenkan_cross_above = False
            tenkan_cross_below = False
        
        if position == 0:
            # Flat - look for entry: Kumo twist in direction of 1w trend with volume spike
            # Long: Tenkan crosses above Kijun AND price > cloud_bottom AND 1w bullish (price > Senkou Span B_1w) AND volume spike
            # Short: Tenkan crosses below Kijun AND price < cloud_top AND 1w bearish (price < Senkou Span B_1w) AND volume spike
            bullish_1w = close_val > span_b_1w
            bearish_1w = close_val < span_b_1w
            price_above_cloud = close_val > cloud_bottom
            price_below_cloud = close_val < cloud_top
            
            if tenkan_cross_above and price_above_cloud and bullish_1w and vol_spike:
                signals[i] = size
                position = 1
            elif tenkan_cross_below and price_below_cloud and bearish_1w and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit when: Tenkan crosses below Kijun (momentum shift) OR price falls below cloud bottom
            if tenkan_cross_below or close_val < cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when: Tenkan crosses above Kijun (momentum shift) OR price rises above cloud top
            if tenkan_cross_above or close_val > cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_VolumeSpike_HTF"
timeframe = "6h"
leverage = 1.0