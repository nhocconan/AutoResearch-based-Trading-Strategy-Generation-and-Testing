#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Ichimoku components (9, 26, 52) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    tenkan_sen = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_tenkan - 1:
            tenkan_sen[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                             np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    kijun_sen = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_kijun - 1:
            kijun_sen[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                            np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    senkou_span_b = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= period_senkou_b - 1:
            senkou_span_b[i] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                                np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    
    # Align Ichimoku components to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # === 6h Volume spike filter ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    volume_spike = volume > vol_ma_20 * 2.0  # Volume at least 2x 20-period average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Ichimoku cloud: green when Senkou Span A > Senkou Span B, red otherwise
        # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
        cloud_top = np.maximum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = np.minimum(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # TK Cross: Tenkan-sen crosses Kijun-sen
        tk_cross_up = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tk_cross_down = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        vol_confirm = volume_spike[i] if not np.isnan(volume_spike[i]) else False
        
        # Entry logic
        if position == 0:
            # Long: TK cross up + price above cloud + volume spike
            if tk_cross_up and price_above_cloud and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: TK cross down + price below cloud + volume spike
            elif tk_cross_down and price_below_cloud and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: TK cross down OR price falls below cloud
            if tk_cross_down or price_below_cloud:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TK cross up OR price rises above cloud
            if tk_cross_up or price_above_cloud:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "Ichimoku_TK_Cross_Cloud_VolumeSpike"
timeframe = "6h"
leverage = 1.0