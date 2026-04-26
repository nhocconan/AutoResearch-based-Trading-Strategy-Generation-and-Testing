#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_Filter
Hypothesis: 6h Ichimoku Kumo twist (Senkou Span A/B cross) with 1w trend filter (price vs Kumo) and volume confirmation.
Enters long when Senkou Span A crosses above Senkou Span B (bullish Kumo twist) with price above Kumo and volume spike.
Enters short when Senkou Span A crosses below Senkou Span B (bearish Kumo twist) with price below Kumo and volume spike.
Exits on opposite Kumo twist or when price re-enters Kumo.
Designed for 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
Uses discrete position sizing (0.25) to minimize churn. Works in both bull and bear markets by following 1w trend via Kumo position.
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
    
    # Calculate Ichimoku components on 6h timeframe
    # Conversion Line (Tenkan-sen): (9-period high + low)/2
    period_9 = 9
    max_high_9 = pd.Series(high).rolling(window=period_9, center=False).max().values
    min_low_9 = pd.Series(low).rolling(window=period_9, center=False).min().values
    tenkan_sen = (max_high_9 + min_low_9) / 2.0
    
    # Base Line (Kijun-sen): (26-period high + low)/2
    period_26 = 26
    max_high_26 = pd.Series(high).rolling(window=period_26, center=False).max().values
    min_low_26 = pd.Series(low).rolling(window=period_26, center=False).min().values
    kijun_sen = (max_high_26 + min_low_26) / 2.0
    
    # Leading Span A (Senkou Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0)
    
    # Leading Span B (Senkou Span B): (52-period high + low)/2 shifted 26 periods ahead
    period_52 = 52
    max_high_52 = pd.Series(high).rolling(window=period_52, center=False).max().values
    min_low_52 = pd.Series(low).rolling(window=period_52, center=False).min().values
    senkou_span_b = ((max_high_52 + min_low_52) / 2.0)
    
    # Lagging Span (Chikou Span): close shifted 26 periods behind (not used for signals)
    
    # Kumo twist signals: Senkou Span A cross above/below Senkou Span B
    # Bullish twist: Senkou Span A crosses above Senkou Span B
    # Bearish twist: Senkou Span A crosses below Senkou Span B
    bullish_twist = (senkou_span_a > senkou_span_b) & (np.roll(senkou_span_a, 1) <= np.roll(senkou_span_b, 1))
    bearish_twist = (senkou_span_a < senkou_span_b) & (np.roll(senkou_span_a, 1) >= np.roll(senkou_span_b, 1))
    
    # Price relative to Kumo (cloud)
    price_above_kumo = (close > np.maximum(senkou_span_a, senkou_span_b))
    price_below_kumo = (close < np.minimum(senkou_span_a, senkou_span_b))
    price_in_kumo = ~(price_above_kumo | price_below_kumo)
    
    # Load 1w data for trend filter: price vs Kumo on weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku on 1w for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Tenkan-sen and Kijun-sen
    max_high_9_1w = pd.Series(high_1w).rolling(window=9, center=False).max().values
    min_low_9_1w = pd.Series(low_1w).rolling(window=9, center=False).min().values
    tenkan_sen_1w = (max_high_9_1w + min_low_9_1w) / 2.0
    
    max_high_26_1w = pd.Series(high_1w).rolling(window=26, center=False).max().values
    min_low_26_1w = pd.Series(low_1w).rolling(window=26, center=False).min().values
    kijun_sen_1w = (max_high_26_1w + min_low_26_1w) / 2.0
    
    # 1w Senkou Span A and B
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2.0)
    max_high_52_1w = pd.Series(high_1w).rolling(window=52, center=False).max().values
    min_low_52_1w = pd.Series(low_1w).rolling(window=52, center=False).min().values
    senkou_span_b_1w = ((max_high_52_1w + min_low_52_1w) / 2.0)
    
    # 1w Kumo trend: price above/below Kumo
    price_above_kumo_1w = (close_1w > np.maximum(senkou_span_a_1w, senkou_span_b_1w))
    price_below_kumo_1w = (close_1w < np.minimum(senkou_span_a_1w, senkou_span_b_1w))
    
    # Align 1w Kumo trend to 6h
    price_above_kumo_1w_aligned = align_htf_to_ltf(prices, df_1w, price_above_kumo_1w.astype(float))
    price_below_kumo_1w_aligned = align_htf_to_ltf(prices, df_1w, price_below_kumo_1w.astype(float))
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period for Senkou Span B)
    start_idx = 52 + 26  # 52 for calculation + 26 for shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(senkou_span_a[i]) or np.isnan(senkou_span_b[i]) or 
            np.isnan(price_above_kumo_1w_aligned[i]) or np.isnan(price_below_kumo_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: bullish Kumo twist + price above Kumo + 1w bullish trend + volume spike
        if bullish_twist[i] and price_above_kumo[i] and price_above_kumo_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish Kumo twist + price below Kumo + 1w bearish trend + volume spike
        elif bearish_twist[i] and price_below_kumo[i] and price_below_kumo_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite Kumo twist or price re-enters Kumo
        elif position == 1 and (bearish_twist[i] or price_in_kumo[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (bullish_twist[i] or price_in_kumo[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0