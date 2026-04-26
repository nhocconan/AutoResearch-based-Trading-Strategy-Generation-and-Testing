#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Breakout_1dTrend_v1
Hypothesis: Trade Ichimoku cloud breaks on 6h with 1d EMA50 trend filter and volume confirmation (1.8x median). Only trade in direction of 1d EMA50 trend to reduce whipsaws. Target: 12-25 trades/year on 6h. Works in bull/bear by adapting to trend and using cloud as dynamic support/resistance.
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
    
    # Get 1d data for HTF trend and Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Ichimoku components on 6h: Tenkan-sen (9), Kijun-sen (26), Senkou Span A/B (52 displacement)
    period_tenkan = 9
    period_kijun = 26
    period_senkou_b = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, displaced 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, displaced 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values + 
                      pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values) / 2)
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: 1.8x median volume (24-period) for signal
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d, Ichimoku calculations
    start_idx = max(50, period_senkou_b + displacement, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        # Cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        lower_cloud = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        if position == 0:
            # Long: break above cloud with volume confirmation, and uptrend
            long_signal = (close_val > upper_cloud) and \
                          (volume_val > 1.8 * vol_median_val) and \
                          uptrend
            
            # Short: break below cloud with volume confirmation, and downtrend
            short_signal = (close_val < lower_cloud) and \
                           (volume_val > 1.8 * vol_median_val) and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long until price closes back below cloud
            signals[i] = 0.25
            if close_val < upper_cloud:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short until price closes back above cloud
            signals[i] = -0.25
            if close_val > lower_cloud:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Breakout_1dTrend_v1"
timeframe = "6h"
leverage = 1.0