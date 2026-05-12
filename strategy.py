#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Breakout_Trend_Filter
Uses 1d Ichimoku Cloud (Tenkan/Kijun/Senkou A/B) as trend filter and support/resistance on 6h timeframe.
Enters long when price breaks above Senkou Span A (upper cloud) with bullish TK cross and price above cloud.
Enters short when price breaks below Senkou Span B (lower cloud) with bearish TK cross and price below cloud.
Uses 1d ADX > 25 to filter for trending conditions only, avoiding ranging markets.
Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drain.
Ichimoku provides dynamic support/resistance that adapts to volatility, working in both bull and bear markets.
"""

name = "6h_1d_Ichimoku_Breakout_Trend_Filter"
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
    volume = prices['volume'].values
    
    # Daily data for Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku calculations (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)  # Shifted 26 periods ahead
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2).shift(26)
    
    # ADX calculation (14-period)
    # +DM, -DM, TR
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), np.abs(low_1d[1:] - low_1d[:-1]))
    )
    # Pad first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(tenkan_sen_aligned[i]) or
            np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or
            np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Bullish TK cross: Tenkan > Kijun
        tk_bullish = tenkan_sen_aligned[i] > kijun_sen_aligned[i]
        # Bearish TK cross: Tenkan < Kijun
        tk_bearish = tenkan_sen_aligned[i] < kijun_sen_aligned[i]
        # Price above cloud: close > Senkou Span A AND close > Senkou Span B
        price_above_cloud = close[i] > senkou_span_a_aligned[i] and close[i] > senkou_span_b_aligned[i]
        # Price below cloud: close < Senkou Span A AND close < Senkou Span B
        price_below_cloud = close[i] < senkou_span_a_aligned[i] and close[i] < senkou_span_b_aligned[i]
        # Trending market: ADX > 25
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # LONG: Price breaks above Senkou Span A (upper cloud) + bullish TK cross + price above cloud + trending
            if (close[i] > senkou_span_a_aligned[i] and 
                tk_bullish and 
                price_above_cloud and 
                trending):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Senkou Span B (lower cloud) + bearish TK cross + price below cloud + trending
            elif (close[i] < senkou_span_b_aligned[i] and 
                  tk_bearish and 
                  price_below_cloud and 
                  trending):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Senkou Span A (cloud bottom) OR bearish TK cross
            if (close[i] < senkou_span_a_aligned[i]) or \
               (not tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Senkou Span B (cloud top) OR bullish TK cross
            if (close[i] > senkou_span_b_aligned[i]) or \
               (tk_bullish):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals