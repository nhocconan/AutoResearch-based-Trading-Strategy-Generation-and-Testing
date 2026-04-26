#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Breakout_1dEMA50_VolumeSpike
Hypothesis: Trade 6h Ichimoku cloud breakouts in direction of 1d EMA50 trend with volume confirmation.
Uses Ichimoku cloud (Senkou Span A/B) from 6h for trend and support/resistance,
1d EMA50 for higher timeframe trend alignment, and volume spike on 6h for breakout conviction.
Designed to capture strong trending moves while avoiding whipsaws in ranging markets.
Target: 12-25 trades/year per symbol (~50-100 total over 4 years) to minimize fee drag.
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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku components on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high_6h).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low_6h).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high_6h).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low_6h).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Shift 26 periods ahead for alignment (will be handled by align_htf_to_ltf)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low_6h).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (max_high_senkou_b + min_low_senkou_b) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Current cloud boundaries (Senkou Span A/B from 26 periods ago, now aligned)
    # Since we shifted Senkou spans ahead in calculation, we use current values
    upper_cloud = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    lower_cloud = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Volume confirmation: volume > 2.0x 20-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), Ichimoku (52), volume MA (20)
    start_idx = max(50, 52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(tenkan_sen_aligned[i]) or 
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(upper_cloud[i]) or
            np.isnan(lower_cloud[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Ichimoku signals
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        tenkan_above_kijun = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        tenkan_below_kijun = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud + price above 1d EMA50 + Tenkan > Kijun + volume spike
            long_breakout = price_above_cloud
            long_signal = long_breakout and price_above_ema and tenkan_above_kijun and volume_spike[i]
            
            # Short: price breaks below cloud + price below 1d EMA50 + Tenkan < Kijun + volume spike
            short_breakout = price_below_cloud
            short_signal = short_breakout and price_below_ema and tenkan_below_kijun and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below cloud OR Tenkan < Kijun OR trend turns bearish
            if (close[i] < lower_cloud[i] or tenkan_below_kijun or not price_above_ema):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above cloud OR Tenkan > Kijun OR trend turns bullish
            if (close[i] > upper_cloud[i] or tenkan_above_kijun or not price_below_ema):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_Breakout_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0