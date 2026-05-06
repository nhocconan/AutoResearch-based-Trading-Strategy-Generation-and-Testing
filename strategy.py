#!/usr/bin/env python3
# 6h_1dIchimoku_Cloud_TK_Cross_Trend_Filter
# Uses daily Ichimoku cloud and TK cross for trend direction with price relative to cloud.
# Long when price above cloud and TK cross bullish, short when price below cloud and TK cross bearish.
# Cloud acts as dynamic support/resistance, working in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "6h_1dIchimoku_Cloud_TK_Cross_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max() + 
                  pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max() + 
                 pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b = (pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max() + 
                     pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_6h = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_6h = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_6h, senkou_span_b_6h)
    cloud_bottom = np.minimum(senkou_span_a_6h, senkou_span_b_6h)
    
    # TK Cross: Tenkan-sen > Kijun-sen (bullish) or < (bearish)
    tk_cross_bullish = tenkan_sen_6h > kijun_sen_6h
    tk_cross_bearish = tenkan_sen_6h < kijun_sen_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tk_cross_bullish[i]) or np.isnan(tk_cross_bearish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above cloud and TK cross bullish
            if close[i] > cloud_top[i] and tk_cross_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below cloud and TK cross bearish
            elif close[i] < cloud_bottom[i] and tk_cross_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price drops below cloud bottom or TK cross turns bearish
            if close[i] < cloud_bottom[i] or not tk_cross_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above cloud top or TK cross turns bullish
            if close[i] > cloud_top[i] or not tk_cross_bearish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals