#!/usr/bin/env python3
"""
6h Ichimoku Cloud + TK Cross with 1d Trend Filter and Volume Spike Confirmation
Hypothesis: Ichimoku cloud (from 1d) provides dynamic support/resistance. TK cross (Tenkan/Kijun) on 6h signals momentum, with cloud acting as filter: long only when price above cloud (bullish regime), short only when below cloud (bearish regime). Volume spike (>2.0x 20-bar MA) confirms breakout strength. Works in bull markets via upside breaks above cloud and in bear markets via downside breaks below cloud. Discrete sizing (0.25) limits fee drag. Target: 50-150 trades over 4 years (12-37/year).
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
    
    # Get 1d data for Ichimoku cloud (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full(len(df_1d), np.nan)
    for i in range(period_tenkan - 1, len(df_1d)):
        tenkan_sen[i] = (np.max(high_1d[i-period_tenkan+1:i+1]) + np.min(low_1d[i-period_tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full(len(df_1d), np.nan)
    for i in range(period_kijun - 1, len(df_1d)):
        kijun_sen[i] = (np.max(high_1d[i-period_kijun+1:i+1]) + np.min(low_1d[i-period_kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + 26
            if idx < len(df_1d):
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    senkou_span_b = np.full(len(df_1d), np.nan)
    for i in range(period_senkou_b - 1, len(df_1d)):
        idx = i + 26
        if idx < len(df_1d):
            senkou_span_b[idx] = (np.max(high_1d[i-period_senkou_b+1:i+1]) + np.min(low_1d[i-period_senkou_b+1:i+1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku (52 + 26 for Senkou) and volume MA
    start_idx = max(52 + 26, 20)  # 78 for Ichimoku, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Cloud boundaries: top and bottom of cloud
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # TK cross: Tenkan crosses above/below Kijun
        tk_cross_above = tenkan_val > kijun_val
        tk_cross_below = tenkan_val < kijun_val
        
        # Price relative to cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price above cloud + TK bullish cross (Tenkan > Kijun) + volume confirmation
            long_signal = price_above_cloud and tk_cross_above and volume_confirm
            # Short: price below cloud + TK bearish cross (Tenkan < Kijun) + volume confirmation
            short_signal = price_below_cloud and tk_cross_below and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below cloud OR TK bearish cross
            if (curr_close < cloud_top) or (tenkan_val < kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above cloud OR TK bullish cross
            if (curr_close > cloud_bottom) or (tenkan_val > kijun_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TK_Cross_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0