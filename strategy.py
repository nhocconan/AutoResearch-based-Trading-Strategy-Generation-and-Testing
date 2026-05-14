#!/usr/bin/env python3
# Hypothesis: 6h Ichimoku Cloud (TK Cross) with 1d trend filter and volume confirmation.
# Uses Kumo (cloud) from 1d as trend filter, TK cross on 6h for entry, volume spike for confirmation.
# Designed to capture trend continuation in both bull and bear markets by aligning 6h momentum with 1d Ichimoku structure.
# Targets 50-150 total trades over 4 years.

name = "6h_Ichimoku_TK_Cross_1dKumoTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components: Tenkan-sen, Kijun-sen, Senkou Span A/B"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over past tenkan periods
    tenkan_sen = (pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max() + 
                  pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min()) / 2
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over past kijun periods
    kijun_sen = (pd.Series(high).rolling(window=kijun, min_periods=kijun).max() + 
                 pd.Series(low).rolling(window=kijun, min_periods=kijun).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted forward kijun periods
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over past senkou periods shifted forward kijun
    senkou_span_b = ((pd.Series(high).rolling(window=senkou, min_periods=senkou).max() + 
                      pd.Series(low).rolling(window=senkou, min_periods=senkou).min()) / 2).shift(kijun)
    # Chikou Span (Lagging Span): close shifted back kijun periods (not used for signals)
    chikou_span = pd.Series(close).shift(-kijun)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # need sufficient data for Ichimoku
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    # TK Cross: Tenkan > Kijun (bullish), Tenkan < Kijun (bearish)
    tk_cross_bull = tenkan > kijun
    tk_cross_bear = tenkan < kijun
    # Price above/below cloud
    price_above_cloud = (close > senkou_a) & (close > senkou_b)
    price_below_cloud = (close < senkou_a) & (close < senkou_b)
    
    # Volume spike: > 1.8x 20-period average (stricter threshold for lower frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Ichimoku Cloud (trend filter)
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    # Kumo twist: Senkou A > Senkou B = bullish cloud, Senkou A < Senkou B = bearish cloud
    kumo_bullish = senkou_a_1d > senkou_b_1d
    kumo_bearish = senkou_a_1d < senkou_b_1d
    # Align to 6h timeframe
    kumo_bullish_aligned = align_htf_to_ltf(prices, df_1d, kumo_bullish.astype(float))
    kumo_bearish_aligned = align_htf_to_ltf(prices, df_1d, kumo_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(kumo_bullish_aligned[i]) or np.isnan(kumo_bearish_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above cloud AND TK cross bullish AND 1d Kumo bullish AND volume spike
            if (price_above_cloud[i] and 
                tk_cross_bull[i] and 
                kumo_bullish_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below cloud AND TK cross bearish AND 1d Kumo bearish AND volume spike
            elif (price_below_cloud[i] and 
                  tk_cross_bear[i] and 
                  kumo_bearish_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below cloud OR TK cross turns bearish
            if (not price_above_cloud[i]) or (not tk_cross_bull[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above cloud OR TK cross turns bullish
            if (not price_below_cloud[i]) or (not tk_cross_bear[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals