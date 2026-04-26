#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, enter long when price breaks above Kumo cloud AND Tenkan/Kijun cross bullish AND 1d trend up (close > EMA50) AND volume > 1.5x 20-period average. Enter short when price breaks below Kumo cloud AND Tenkan/Kijun cross bearish AND 1d trend down (close < EMA50) AND volume spike. Uses Kumo twist (senkou span A/B cross) as trend confirmation filter to avoid false breakouts. Designed for low-moderate trade frequency (12-30/year) with edge in both bull and bear markets via trend alignment and volatility-based entry. Ichimoku provides dynamic support/resistance and trend strength in one system.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span"""
    n = len(high)
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    tenkan_sen = np.full(n, np.nan)
    for i in range(tenkan-1, n):
        tenkan_sen[i] = (np.max(high[i-tenkan+1:i+1]) + np.min(low[i-tenkan+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    kijun_sen = np.full(n, np.nan)
    for i in range(kijun-1, n):
        kijun_sen[i] = (np.max(high[i-kijun+1:i+1]) + np.min(low[i-kijun+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
    senkou_span_a = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
            idx = i + kijun  # plotted 26 periods ahead
            if idx < n:
                senkou_span_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods plotted 26 periods ahead
    senkou_span_b = np.full(n, np.nan)
    for i in range(senkou-1, n):
        idx = i + kijun  # plotted 26 periods ahead
        if idx < n:
            senkou_span_b[idx] = (np.max(high[i-senkou+1:i+1]) + np.min(low[i-senkou+1:i+1])) / 2
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind
    chikou_span = np.full(n, np.nan)
    for i in range(n - kijun):
        chikou_span[i] = close[i + kijun]
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku on 6h data
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # Kumo Twist: Senkou Span A/B cross (trend strength confirmation)
    # Bullish twist: Senkou A > Senkou B
    # Bearish twist: Senkou A < Senkou B
    kumo_twist_bullish = senkou_a_6h > senkou_b_6h
    kumo_twist_bearish = senkou_a_6h < senkou_b_6h
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku warmup (52), EMA warmup (50), volume MA warmup (20)
    start_idx = max(52, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or np.isnan(senkou_a_6h[i]) or 
            np.isnan(senkou_b_6h[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Ichimoku signals
        price_above_kumo = close[i] > max(senkou_a_6h[i], senkou_b_6h[i])
        price_below_kumo = close[i] < min(senkou_a_6h[i], senkou_b_6h[i])
        tenkan_cross_above_kijun = tenkan_6h[i] > kijun_6h[i]
        tenkan_cross_below_kijun = tenkan_6h[i] < kijun_6h[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above Kumo AND Tenkan/Kijun bullish cross AND 1d uptrend AND volume spike AND Kumo bullish twist
            long_signal = (price_above_kumo and tenkan_cross_above_kijun and 
                          trend_uptrend and volume_spike[i] and kumo_twist_bullish[i])
            
            # Short: price below Kumo AND Tenkan/Kijun bearish cross AND 1d downtrend AND volume spike AND Kumo bearish twist
            short_signal = (price_below_kumo and tenkan_cross_below_kijun and 
                           trend_downtrend and volume_spike[i] and kumo_twist_bearish[i])
            
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
            # Exit: price breaks below Kumo OR Tenkan/Kijun cross bearish OR trend change to downtrend OR Kumo bearish twist
            if (price_below_kumo or tenkan_cross_below_kijun or not trend_uptrend or not kumo_twist_bullish[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Kumo OR Tenkan/Kijun cross bullish OR trend change to uptrend OR Kumo bullish twist
            if (price_above_kumo or tenkan_cross_above_kijun or not trend_downtrend or not kumo_twist_bearish[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Breakout_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0