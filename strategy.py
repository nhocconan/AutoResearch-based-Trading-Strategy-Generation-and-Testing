#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter
Hypothesis: Use Ichimoku cloud twist (Senkou Span A/B cross) as trend change signal, confirmed by 1d EMA50 trend and volume spike.
Enters long when: price above cloud, Senkou Span A crosses above Senkou Span B (bullish twist), 1d EMA50 rising, volume > 2.0 * 20-period average.
Enters short when: price below cloud, Senkou Span A crosses below Senkou Span B (bearish twist), 1d EMA50 falling, volume > 2.0 * 20-period average.
Exits when: price crosses opposite Senkou Span (A for long exit, B for short exit) or cloud reverses.
Uses discrete 0.25 position size. Targets 12-25 trades/year to avoid fee drag on 6h timeframe.
Works in both bull/bear: cloud acts as dynamic support/resistance, twist catches early trend changes, volume filter ensures commitment.
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
    
    # Ichimoku parameters (standard: 9, 26, 52)
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26  # Kumo displacement
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past 9 periods
    tenkan_sen = (pd.Series(high).rolling(window=tenkan_period, min_periods=tenkan_period).max().values +
                  pd.Series(low).rolling(window=tenkan_period, min_periods=tenkan_period).min().values) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past 26 periods
    kijun_sen = (pd.Series(high).rolling(window=kijun_period, min_periods=kijun_period).max().values +
                 pd.Series(low).rolling(window=kijun_period, min_periods=kijun_period).min().values) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past 52 periods, plotted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max().values +
                      pd.Series(low).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min().values) / 2)
    
    # Cloud twist detection: Senkou Span A crossing Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    ss_a_above_ss_b = senkou_span_a > senkou_span_b
    ss_a_below_ss_b = senkou_span_a < senkou_span_b
    bullish_twist = ss_a_above_ss_b & ~np.roll(ss_a_above_ss_b, 1)
    bearish_twist = ss_a_below_ss_b & ~np.roll(ss_a_below_ss_b, 1)
    # Handle first element
    bullish_twist[0] = False
    bearish_twist[0] = False
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need max(52 for Senkou B, 20 for volume, 50 for 1d EMA)
    start_idx = max(52, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for Ichimoku cloud twist with trend and volume confirmation
            # Long: bullish twist + price above cloud + 1d EMA50 uptrend + volume spike
            price_above_cloud = close_val > max(senkou_span_a[i], senkou_span_b[i])
            ema_rising = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            long_entry = bullish_twist[i] and price_above_cloud and ema_rising and volume_spike[i]
            
            # Short: bearish twist + price below cloud + 1d EMA50 downtrend + volume spike
            price_below_cloud = close_val < min(senkou_span_a[i], senkou_span_b[i])
            ema_falling = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
            short_entry = bearish_twist[i] and price_below_cloud and ema_falling and volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price crosses below Senkou Span A (cloud support) or cloud becomes bearish
            if (close_val < senkou_span_a[i]) or (senkou_span_a[i] < senkou_span_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price crosses above Senkou Span B (cloud resistance) or cloud becomes bullish
            if (close_val > senkou_span_b[i]) or (senkou_span_a[i] > senkou_span_b[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0