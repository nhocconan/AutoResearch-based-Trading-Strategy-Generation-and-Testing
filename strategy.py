#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_v1
Hypothesis: On 6h timeframe, Ichimoku cloud twist (Senkou Span A/B cross) with 1d trend filter (price vs Kumo) and volume confirmation captures major trend reversals in both bull and bear markets. The cloud acts as dynamic support/resistance, reducing whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1d data ONCE before loop for HTF Ichimoku and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = np.concatenate([np.full(26, np.nan), close_1d[:-26]]) if len(close_1d) >= 26 else np.full_like(close_1d, np.nan)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b)
    chikou_span_6h = align_htf_to_ltf(prices, df_1d, chikou_span)
    
    # Kumo (cloud) twist: Senkou Span A crosses Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    senkou_a_above_b = senkou_span_a_6h > senkou_span_b_6h
    senkou_a_below_b = senkou_span_a_6h < senkou_span_b_6h
    
    # Detect twists (crossovers)
    bullish_twist = senkou_a_above_b & ~np.concatenate([[False], senkou_a_above_b[:-1]])
    bearish_twist = senkou_a_below_b & ~np.concatenate([[False], senkou_a_below_b[:-1]])
    
    # 1d trend filter: price above/below cloud
    price_above_cloud = (close > np.maximum(senkou_span_a_6h, senkou_span_b_6h))
    price_below_cloud = (close < np.minimum(senkou_span_a_6h, senkou_span_b_6h))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 52 for Senkou B, 20 for volume MA)
    start_idx = max(52, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_sen_6h[i]) or 
            np.isnan(kijun_sen_6h[i]) or
            np.isnan(senkou_span_a_6h[i]) or
            np.isnan(senkou_span_b_6h[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long logic: bullish Kumo twist + price above cloud + volume spike
        if bullish_twist[i] and price_above_cloud[i] and volume_spike[i]:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: bearish Kumo twist + price below cloud + volume spike
        elif bearish_twist[i] and price_below_cloud[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: opposite twist OR price enters cloud (Kumo break)
        elif position == 1 and (bearish_twist[i] or not price_above_cloud[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_twist[i] or not price_below_cloud[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_v1"
timeframe = "6h"
leverage = 1.0