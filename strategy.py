#!/usr/bin/env python3
"""
4h_IchimokuKumo_CryptoTrend
Hypothesis: Ichimoku Kinko Hyo system with Kumo (cloud) filter on 4h timeframe, 
combined with 12h trend filter and volume confirmation for high-probability entries.
Works in bull markets by buying pullbacks to Kumo support in uptrends, 
and in bear markets by selling rallies to Kumo resistance in downtrends.
Kumo acts as dynamic support/resistance, reducing whipsaws in sideways markets.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
"""

name = "4h_IchimokuKumo_CryptoTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Ichimoku components on 4h (Tenkan-sen, Kijun-sen, Senkou Span A/B)
    period_tenkan = 9
    period_kijun = 26
    period_senkou = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = np.full(n, np.nan)
    for i in range(period_tenkan - 1, n):
        tenkan_sen[i] = (np.max(high[i - period_tenkan + 1:i + 1]) + 
                         np.min(low[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = np.full(n, np.nan)
    for i in range(period_kijun - 1, n):
        kijun_sen[i] = (np.max(high[i - period_kijun + 1:i + 1]) + 
                        np.min(low[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, shifted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + period_kijun
            if idx < n:
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2, shifted 26 periods ahead
    senkou_span_b = np.full(n, np.nan)
    for i in range(period_senkou - 1, n):
        idx = i + period_kijun
        if idx < n:
            senkou_span_b[idx] = (np.max(high[i - period_senkou + 1:i + 1]) + 
                                  np.min(low[i - period_senkou + 1:i + 1])) / 2
    
    # Kumo (Cloud) top and bottom
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume SMA20 for volume confirmation
    volume_12h = df_12h['volume'].values
    vol_sma20_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_sma20_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_sma20_12h[i] = (vol_sma20_12h[i-1] * 19 + volume_12h[i]) / 20
    vol_sma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period_kijun + period_senkou, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or \
           np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x average 12h volume (scaled)
        vol_12h_scaled = vol_sma20_12h_aligned[i] / 3.0  # 3x 4h periods in 12h
        volume_confirm = volume[i] > 1.3 * vol_12h_scaled
        
        if position == 0:
            # Long: Price above Kumo in uptrend with volume confirmation
            if (close[i] > kumo_top[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price below Kumo in downtrend with volume confirmation
            elif (close[i] < kumo_bottom[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls below Kumo base or trend reversal
            if (close[i] < kumo_bottom[i] or 
                close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above Kumo top or trend reversal
            if (close[i] > kumo_top[i] or 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals