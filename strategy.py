#!/usr/bin/env python3
"""
4h_Ichimoku_Cloud_Twist_WeeklyTrend
Hypothesis: Ichimoku Cloud twist (Tenkan/Kijun cross) on 4h with weekly trend filter and volume spike confirmation.
Trades in direction of higher timeframe trend with cloud twist as entry signal.
Targets 20-30 trades/year to minimize fee drift while capturing trend changes.
"""

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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 26:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen_1w = (pd.Series(high_1w).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low_1w).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen_1w = (pd.Series(high_1w).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low_1w).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    senkou_span_a_1w = ((tenkan_sen_1w + kijun_sen_1w) / 2).shift(26)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 52 periods
    senkou_span_b_1w = ((pd.Series(high_1w).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low_1w).rolling(window=52, min_periods=52).min()) / 2).shift(52)
    
    # Align weekly Ichimoku to 4h
    tenkan_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_1w.values)
    kijun_sen_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_1w.values)
    senkou_span_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_1w.values)
    senkou_span_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_1w.values)
    
    # Calculate 4h Ichimoku for entry signal (cloud twist)
    tenkan_sen_4h = (pd.Series(high).rolling(window=9, min_periods=9).max() + 
                     pd.Series(low).rolling(window=9, min_periods=9).min()) / 2
    kijun_sen_4h = (pd.Series(high).rolling(window=26, min_periods=26).max() + 
                    pd.Series(low).rolling(window=26, min_periods=26).min()) / 2
    # 4h Senkou Span A and B
    senkou_span_a_4h = ((tenkan_sen_4h + kijun_sen_4h) / 2).shift(26)
    senkou_span_b_4h = ((pd.Series(high).rolling(window=52, min_periods=52).max() + 
                         pd.Series(low).rolling(window=52, min_periods=52).min()) / 2).shift(52)
    
    # Volume confirmation: >1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_1w_aligned[i]) or np.isnan(kijun_sen_1w_aligned[i]) or
            np.isnan(senkou_span_a_1w_aligned[i]) or np.isnan(senkou_span_b_1w_aligned[i]) or
            np.isnan(tenkan_sen_4h[i]) or np.isnan(kijun_sen_4h[i]) or
            np.isnan(senkou_span_a_4h[i]) or np.isnan(senkou_span_b_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/both span lines
        weekly_bullish = (close[i] > senkou_span_a_1w_aligned[i] and 
                         close[i] > senkou_span_b_1w_aligned[i])
        weekly_bearish = (close[i] < senkou_span_a_1w_aligned[i] and 
                         close[i] < senkou_span_b_1w_aligned[i])
        
        # 4h cloud twist: Tenkan/Kijun cross
        tk_cross_up = (tenkan_sen_4h[i-1] <= kijun_sen_4h[i-1] and 
                      tenkan_sen_4h[i] > kijun_sen_4h[i])
        tk_cross_down = (tenkan_sen_4h[i-1] >= kijun_sen_4h[i-1] and 
                        tenkan_sen_4h[i] < kijun_sen_4h[i])
        
        # Price relative to cloud
        above_cloud = (close[i] > senkou_span_a_4h[i] and 
                      close[i] > senkou_span_b_4h[i])
        below_cloud = (close[i] < senkou_span_a_4h[i] and 
                      close[i] < senkou_span_b_4h[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Entry logic: cloud twist in direction of weekly trend
        long_entry = vol_confirm and weekly_bullish and tk_cross_up and above_cloud
        short_entry = vol_confirm and weekly_bearish and tk_cross_down and below_cloud
        
        # Exit logic: opposite twist or trend change
        long_exit = (tk_cross_down and below_cloud) or (not weekly_bullish)
        short_exit = (tk_cross_up and above_cloud) or (not weekly_bearish)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Ichimoku_Cloud_Twist_WeeklyTrend"
timeframe = "4h"
leverage = 1.0