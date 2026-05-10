#!/usr/bin/env python3
"""
6H_Ichimoku_Kumo_Twist_1wTrend_Filter
Hypothesis: Ichimoku Kumo twist (Tenkan-Kijun cross) with weekly trend filter and volume confirmation.
Works in bull/bear by only taking trades in direction of weekly trend (using weekly close > weekly open).
Kumo acts as dynamic support/resistance. Entry on TK cross in direction of trend, with price outside Kumo.
Targets low trade frequency (15-30/year) by requiring weekly alignment and Kumo filter.
"""

name = "6H_Ichimoku_Kumo_Twist_1wTrend_Filter"
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
    
    # Get weekly data for trend filter (weekly close > weekly open = bullish)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_bullish = (df_1w['close'] > df_1w['open']).astype(int)
    weekly_bearish = (df_1w['close'] < df_1w['open']).astype(int)
    weekly_trend = weekly_bullish - weekly_bearish  # 1 for bull, -1 for bear, 0 for doji
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.values)
    
    # Get daily data for Ichimoku components (needed for Tenkan, Kijun, Senkou Span)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku calculations (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min()
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min()
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).shift(26)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Kumo (cloud) boundaries: Senkou Span A and Senkou Span B
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    kumo_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    kumo_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20)  # Warmup for Ichimoku and volume
    
    for i in range(start_idx, n):
        if np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]) or \
           np.isnan(weekly_trend_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TK cross: Tenkan crosses above/below Kijun
        tk_cross_up = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_down = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price relative to Kumo
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        
        if position == 0:
            # Long entry: TK cross up + price above Kumo + weekly bullish + volume spike
            if (tk_cross_up and price_above_kumo and 
                weekly_trend_aligned[i] > 0 and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross down + price below Kumo + weekly bearish + volume spike
            elif (tk_cross_down and price_below_kumo and 
                  weekly_trend_aligned[i] < 0 and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TK cross down OR price enters Kumo OR weekly turns bearish
            if (tk_cross_down or not price_above_kumo or weekly_trend_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TK cross up OR price enters Kumo OR weekly turns bullish
            if (tk_cross_up or not price_below_kumo or weekly_trend_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals