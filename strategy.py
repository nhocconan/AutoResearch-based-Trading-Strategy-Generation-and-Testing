#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeSpike
Hypothesis: Ichimoku cloud breakout with 12h EMA200 trend filter and volume confirmation (>1.5x 20-bar MA). 
The Ichimoku cloud acts as dynamic support/resistance, and breaks above/below the cloud with 
12h trend alignment and volume confirmation capture strong momentum moves. Designed for 6h timeframe 
to work in both bull and bear markets by following the 12h trend while using Ichimoku structure for 
entries. Volume spike reduces whipsaws in ranging conditions. Target: 12-30 trades/year (50-120 total over 4 years).
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
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # 12h EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Load 1d data for Ichimoku calculation (standard timeframe for Ichimoku)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9   # Conversion Line
    kijun_period = 26   # Base Line
    senkou_span_b_period = 52  # Leading Span B
    displacement = 26   # Kumo displacement
    
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = np.zeros(len(high_1d))
    for i in range(len(tenkan_sen)):
        start_idx = max(0, i - tenkan_period + 1)
        tenkan_sen[i] = (np.max(high_1d[start_idx:i+1]) + np.min(low_1d[start_idx:i+1])) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = np.zeros(len(high_1d))
    for i in range(len(kijun_sen)):
        start_idx = max(0, i - kijun_period + 1)
        kijun_sen[i] = (np.max(high_1d[start_idx:i+1]) + np.min(low_1d[start_idx:i+1])) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward 26 periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    senkou_span_a = np.roll(senkou_span_a, -displacement)
    senkou_span_a[:displacement] = np.nan  # First displacement values are unknown
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted forward 26 periods
    senkou_span_b = np.zeros(len(high_1d))
    for i in range(len(senkou_span_b)):
        start_idx = max(0, i - senkou_span_b_period + 1)
        senkou_span_b[i] = (np.max(high_1d[start_idx:i+1]) + np.min(low_1d[start_idx:i+1])) / 2
    senkou_span_b = np.roll(senkou_span_b, -displacement)
    senkou_span_b[:displacement] = np.nan  # First displacement values are unknown
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: need enough data for Ichimoku (52 + 26 = 78) plus volume MA (20) and 12h EMA (200)
    start_idx = max(78, 20, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_200_12h_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or 
            np.isnan(senkou_span_b_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        ema_200_val = ema_200_12h_aligned[i]
        tenkan_val = tenkan_sen_aligned[i]
        kijun_val = kijun_sen_aligned[i]
        span_a = senkou_span_a_aligned[i]
        span_b = senkou_span_b_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Determine 12h trend: bullish if price > EMA200, bearish if price < EMA200
        bullish_12h = close_val > ema_200_val
        bearish_12h = close_val < ema_200_val
        
        # Entry conditions: price breaks above/below cloud in trend direction with volume spike
        long_entry = (close_val > cloud_top) and bullish_12h and vol_spike
        short_entry = (close_val < cloud_bottom) and bearish_12h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters cloud or trend changes
            if close_val < cloud_top or not bullish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit when price re-enters cloud or trend changes
            if close_val > cloud_bottom or not bearish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0