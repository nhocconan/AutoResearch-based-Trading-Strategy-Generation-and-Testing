#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_Volume
Hypothesis: Ichimoku Cloud on 1d provides trend direction and support/resistance, with weekly trend filter and volume confirmation.
Long when price breaks above cloud in uptrend (weekly trend up) with volume; short when breaks below cloud in downtrend (weekly trend down).
Exit when price re-enters cloud or weekly trend changes. Designed for trend-following in both bull and bear regimes.
Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku Cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 days for Ichimoku
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku Cloud components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = np.full(len(close_1d), np.nan)
    if len(high_1d) >= period_tenkan:
        for i in range(period_tenkan - 1, len(high_1d)):
            tenkan_sen[i] = (np.max(high_1d[i - period_tenkan + 1:i + 1]) + 
                            np.min(low_1d[i - period_tenkan + 1:i + 1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = np.full(len(close_1d), np.nan)
    if len(high_1d) >= period_kijun:
        for i in range(period_kijun - 1, len(high_1d)):
            kijun_sen[i] = (np.max(high_1d[i - period_kijun + 1:i + 1]) + 
                           np.min(low_1d[i - period_kijun + 1:i + 1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods forward
    senkou_span_a = np.full(len(close_1d), np.nan)
    if not np.all(np.isnan(tenkan_sen)) and not np.all(np.isnan(kijun_sen)):
        for i in range(len(close_1d)):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                idx = i + period_kijun  # Shift forward by 26 periods
                if idx < len(close_1d):
                    senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods forward
    period_senkou_b = 52
    senkou_span_b = np.full(len(close_1d), np.nan)
    if len(high_1d) >= period_senkou_b:
        for i in range(period_senkou_b - 1, len(high_1d)):
            idx = i + period_kijun  # Shift forward by 26 periods
            if idx < len(close_1d):
                senkou_span_b[idx] = (np.max(high_1d[i - period_senkou_b + 1:i + 1]) + 
                                     np.min(low_1d[i - period_senkou_b + 1:i + 1])) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Weekly trend filter: EMA(25) on weekly close
    ema_period_w = 25
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period_w:
        ema_1w[ema_period_w - 1] = np.mean(close_1w[:ema_period_w])
        multiplier = 2 / (ema_period_w + 1)
        for i in range(ema_period_w, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align weekly EMA to 6h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (24-period average on 6h)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, ema_period_w, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Cloud boundaries: Senkou Span A and B form the cloud
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Trend filter: price relative to weekly EMA25
        weekly_uptrend = price > ema_1w_aligned[i]
        weekly_downtrend = price < ema_1w_aligned[i]
        
        # Volume confirmation: > 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        if position == 0:
            # Long: price breaks above cloud in weekly uptrend with volume
            if weekly_uptrend and volume_confirmation and price > cloud_top:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud in weekly downtrend with volume
            elif weekly_downtrend and volume_confirmation and price < cloud_bottom:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price re-enters cloud or weekly trend turns down
            if price < cloud_top or price < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price re-enters cloud or weekly trend turns up
            if price > cloud_bottom or price > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0