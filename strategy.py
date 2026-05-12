#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_Filter
Hypothesis: Use Ichimoku cloud from 1d as trend filter and TK cross from 12h for entry signals.
In bull markets, price stays above cloud and TK cross triggers longs; in bear markets, price stays below cloud and TK cross triggers shorts.
Cloud acts as dynamic support/resistance, reducing whipsaws. TK cross provides timely entries.
Targets 12-25 trades/year by requiring both trend alignment and momentum signal.
"""

name = "6h_Ichimoku_Cloud_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku Cloud components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over tenkan period
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over kijun period
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted kijun periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over senkou period shifted kijun periods ahead
    senkou_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                 pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    
    # Chikou Span (Lagging Span): close shifted -kijun periods
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_a.values, senkou_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Ichimoku cloud (trend filter) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    # Calculate Ichimoku on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    chikou_1d_aligned = align_htf_to_ltf(prices, df_1d, chikou_1d)

    # Get 12h data for TK cross (entry signal) ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    # Calculate Tenkan and Kijun for 12h
    tenkan_12h = (pd.Series(df_12h['high']).rolling(window=9, min_periods=9).max() + 
                  pd.Series(df_12h['low']).rolling(window=9, min_periods=9).min()) / 2
    kijun_12h = (pd.Series(df_12h['high']).rolling(window=26, min_periods=26).max() + 
                 pd.Series(df_12h['low']).rolling(window=26, min_periods=26).min()) / 2
    
    # Align 12h TK components to 6h timeframe
    tenkan_12h_aligned = align_htf_to_ltf(prices, df_12h, tenkan_12h.values)
    kijun_12h_aligned = align_htf_to_ltf(prices, df_12h, kijun_12h.values)

    # Volume confirmation: current volume > 1.3x average of last 6 periods
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after warmup
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(tenkan_12h_aligned[i]) or np.isnan(kijun_12h_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine cloud boundaries and trend
        upper_cloud = np.maximum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        lower_cloud = np.minimum(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        
        # Price position relative to cloud
        price_above_cloud = close[i] > upper_cloud
        price_below_cloud = close[i] < lower_cloud
        price_in_cloud = ~(price_above_cloud | price_below_cloud)
        
        # TK cross signals from 12h data
        tk_cross_up = tenkan_12h_aligned[i] > kijun_12h_aligned[i]
        tk_cross_down = tenkan_12h_aligned[i] < kijun_12h_aligned[i]

        if position == 0:
            # LONG: price above cloud + TK cross up + volume
            if price_above_cloud and tk_cross_up and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price below cloud + TK cross down + volume
            elif price_below_cloud and tk_cross_down and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price falls below cloud OR TK cross down
            if price_below_cloud or (not price_above_cloud and tk_cross_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above cloud OR TK cross up
            if price_above_cloud or (not price_below_cloud and tk_cross_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals