#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_1wTrend
# Hypothesis: Ichimoku cloud from daily timeframe provides dynamic support/resistance.
# Tenkan-sen/Kijun-sen cross on 6h triggers entry in direction of weekly trend (above/below weekly Kumo).
# Weekly trend filter avoids counter-trend trades. Cloud acts as dynamic support/resistance for exits.
# Target: 20-40 trades/year to minimize fee drift.

name = "6h_Ichimoku_Cloud_Breakout_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for trend filter
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9_1w = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9_1w = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (high_9_1w + low_9_1w) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26_1w = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26_1w = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun_1w = (high_26_1w + low_26_1w) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1w = ((tenkan_1w + kijun_1w) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52_1w = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52_1w = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (high_52_1w + low_52_1w) / 2
    
    # Chikou Span (Lagging Span): current close shifted 26 periods back
    chikou_1w = df_1w['close'].values
    
    # Align weekly Ichimoku components to 6h
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    chikou_1w_aligned = align_htf_to_ltf(prices, df_1w, chikou_1w)
    
    # Calculate daily Ichimoku components for entry signals
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    high_9_1d = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    high_26_1d = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a_1d = ((tenkan_1d + kijun_1d) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    high_52_1d = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Align daily Ichimoku components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Volume confirmation (24-period MA on 6h = 6 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Ichimoku (52) and volume MA (24)
    start_idx = max(52, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_1w_aligned[i]) or 
            np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or 
            np.isnan(senkou_b_1w_aligned[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or 
            np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(senkou_a_1d_aligned[i]) or 
            np.isnan(senkou_b_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/both Senkou spans = uptrend, below both = downtrend
        weekly_uptrend = (close[i] > senkou_a_1w_aligned[i]) and (close[i] > senkou_b_1w_aligned[i])
        weekly_downtrend = (close[i] < senkou_a_1w_aligned[i]) and (close[i] < senkou_b_1w_aligned[i])
        
        # Daily Ichimoku entry signals: Tenkan/Kijun cross
        tk_cross_bull = tenkan_1d_aligned[i] > kijun_1d_aligned[i]
        tk_cross_bear = tenkan_1d_aligned[i] < kijun_1d_aligned[i]
        
        # Daily cloud: Senkou A/B form support/resistance
        cloud_top = max(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        cloud_bottom = min(senkou_a_1d_aligned[i], senkou_b_1d_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: weekly uptrend + Tenkan crosses above Kijun + price above cloud + volume
            if weekly_uptrend and tk_cross_bull and price_above_cloud and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + Tenkan crosses below Kijun + price below cloud + volume
            elif weekly_downtrend and tk_cross_bear and price_below_cloud and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR price re-enters cloud OR Tenkan crosses below Kijun
            if (not weekly_uptrend) or (not price_above_cloud) or (tenkan_1d_aligned[i] < kijun_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR price re-enters cloud OR Tenkan crosses above Kijun
            if (not weekly_downtrend) or (not price_below_cloud) or (tenkan_1d_aligned[i] > kijun_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals