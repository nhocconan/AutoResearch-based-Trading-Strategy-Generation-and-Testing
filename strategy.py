#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Turn_1wTrend
# Hypothesis: 6-hour Ichimoku Cloud turning points with weekly trend filter. 
# Buy when price closes above cloud in weekly uptrend, sell when price closes below cloud in weekly downtrend.
# Weekly trend filter prevents counter-trend trades; Ichimoku provides objective entry/exit.
# Designed for 6h to achieve 50-150 total trades over 4 years (12-37/year).

name = "6h_Ichimoku_Cloud_Turn_1wTrend"
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
    
    # Weekly data for trend filter and Ichimoku
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Align weekly indicators to 6h timeframe (wait for weekly bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough history for Ichimoku (52 periods)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(tenkan_aligned[i]) or \
           np.isnan(kijun_aligned[i]) or np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Cloud top and bottom
        cloud_top = np.maximum(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = np.minimum(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Determine trend: price above/below cloud + weekly EMA50
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        weekly_uptrend = close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False  # Simplified: use latest weekly close
        
        if position == 0:
            # Long: price closes above cloud in weekly uptrend
            if price_above_cloud and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price closes below cloud in weekly downtrend
            elif price_below_cloud and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below cloud
            if price_below_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above cloud
            if price_above_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals