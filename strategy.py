#!/usr/bin/env python3
"""
6h_1d_1w_Ichimoku_Cloud_Breakout_v1
Hypothesis: Use daily Ichimoku cloud as trend filter and weekly Kumo twist for momentum confirmation. Enter long when price breaks above 6h Kumo top with bullish TK cross and weekly Kumo twist bullish; short when price breaks below 6h Kumo bottom with bearish TK cross and weekly Kumo twist bearish. Ichimoku provides dynamic support/resistance and trend strength, working in both bull and bear markets by adapting to price action. Targets 15-35 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): close shifted back 26 periods
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 6h Ichimoku
    tenkan_6h, kijun_6h, senkou_a_6h, senkou_b_6h, chikou_6h = calculate_ichimoku(high, low, close)
    
    # Kumo (cloud) boundaries
    kumo_top_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    kumo_bottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # TK cross signals
    tk_cross_bullish = tenkan_6h > kijun_6h
    tk_cross_bearish = tenkan_6h < kijun_6h
    
    # Get daily data for Ichimoku (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Daily Kumo
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Daily trend: price above/below Kumo
    price_above_kumo_1d = close_1d > kumo_top_1d
    price_below_kumo_1d = close_1d < kumo_bottom_1d
    
    # Align daily Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    price_above_kumo_1d_aligned = align_htf_to_ltf(prices, df_1d, price_above_kumo_1d.astype(float))
    price_below_kumo_1d_aligned = align_htf_to_ltf(prices, df_1d, price_below_kumo_1d.astype(float))
    
    # Get weekly data for Kumo twist (momentum)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Weekly Kumo twist: Senkou A crossing Senkou B
    kumotwist_bullish_1w = senkou_a_1w > senkou_b_1w
    kumotwist_bearish_1w = senkou_a_1w < senkou_b_1w
    
    # Align weekly Kumo twist to 6h
    kumotwist_bullish_1w_aligned = align_htf_to_ltf(prices, df_1w, kumotwist_bullish_1w.astype(float))
    kumotwist_bearish_1w_aligned = align_htf_to_ltf(prices, df_1w, kumotwist_bearish_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position
    
    for i in range(60, n):  # warmup for Ichimoku
        # Skip if any data not ready
        if (np.isnan(kumo_top_6h[i]) or np.isnan(kumo_bottom_6h[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or
            np.isnan(kumo_top_1d_aligned[i]) or np.isnan(kumo_bottom_1d_aligned[i]) or
            np.isnan(kumotwist_bullish_1w_aligned[i]) or np.isnan(kumotwist_bearish_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bullish conditions: price above 6h Kumo, bullish TK cross, daily price above Kumo, weekly Kumo twist bullish
        long_condition = (close[i] > kumo_top_6h[i] and 
                         tk_cross_bullish[i] and
                         price_above_kumo_1d_aligned[i] > 0.5 and
                         kumotwist_bullish_1w_aligned[i] > 0.5)
        
        # Bearish conditions: price below 6h Kumo, bearish TK cross, daily price below Kumo, weekly Kumo twist bearish
        short_condition = (close[i] < kumo_bottom_6h[i] and 
                          tk_cross_bearish[i] and
                          price_below_kumo_1d_aligned[i] > 0.5 and
                          kumotwist_bearish_1w_aligned[i] > 0.5)
        
        if position == 0:
            if long_condition:
                position = 1
                signals[i] = position_size
            elif short_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price re-enters Kumo or bearish TK cross with weekly bearish twist
            if (close[i] < kumo_top_6h[i] and close[i] > kumo_bottom_6h[i]) or \
               (tk_cross_bearish[i] and kumotwist_bearish_1w_aligned[i] > 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price re-enters Kumo or bullish TK cross with weekly bullish twist
            if (close[i] > kumo_bottom_6h[i] and close[i] < kumo_top_6h[i]) or \
               (tk_cross_bullish[i] and kumotwist_bullish_1w_aligned[i] > 0.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0