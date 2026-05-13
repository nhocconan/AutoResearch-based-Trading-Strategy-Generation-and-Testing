#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Trend_Filtered_Breakout
# Hypothesis: Use 1d Ichimoku cloud as trend filter and 6h price action for breakout signals.
# Long when: 6h price breaks above 6h 20-period high AND price > 1d Kumo (cloud) top AND Tenkan > Kijun (bullish TK cross)
# Short when: 6h price breaks below 6h 20-period low AND price < 1d Kumo (cloud) bottom AND Tenkan < Kijun (bearish TK cross)
# Ichimoku cloud provides dynamic support/resistance and trend direction from higher timeframe.
# TK cross confirms momentum alignment. Breakout of 6h 20-period high/low captures momentum bursts.
# Works in bull (breaks above cloud in uptrend) and bear (breaks below cloud in downtrend).
# Low frequency due to multiple confirmation requirements.

name = "6h_Ichimoku_Cloud_Trend_Filtered_Breakout"
timeframe = "6h"
leverage = 1.0

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

    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    
    # Ichimoku components (9, 26, 52)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).mean() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).mean()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).mean() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).mean()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).mean() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).mean()) / 2).shift(26)
    
    # Kumo (cloud) top and bottom
    kumo_top = np.maximum(senkou_span_a, senkou_span_b)
    kumo_bottom = np.minimum(senkou_span_a, senkou_span_b)
    
    # TK cross signals
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    # Align Ichimoku components to 6h timeframe
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top.values)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom.values)
    tk_cross_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_bullish.values)
    tk_cross_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_cross_bearish.values)
    
    # 6h 20-period high/low for breakout detection
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kumo_top_aligned[i]) or 
            np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(tk_cross_bullish_aligned[i]) or 
            np.isnan(tk_cross_bearish_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 6h 20-period high AND above Kumo top AND bullish TK cross
            if close[i] > high_20[i] and close[i] > kumo_top_aligned[i] and tk_cross_bullish_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 6h 20-period low AND below Kumo bottom AND bearish TK cross
            elif close[i] < low_20[i] and close[i] < kumo_bottom_aligned[i] and tk_cross_bearish_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 6h 20-period low OR bearish TK cross OR price below Kumo bottom
            if close[i] < low_20[i] or tk_cross_bearish_aligned[i] or close[i] < kumo_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 6h 20-period high OR bullish TK cross OR price above Kumo top
            if close[i] > high_20[i] or tk_cross_bullish_aligned[i] or close[i] > kumo_top_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals