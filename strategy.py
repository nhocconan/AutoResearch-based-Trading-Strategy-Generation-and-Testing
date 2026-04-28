#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Trend_WeeklyFilter
Hypothesis: 6-hour Ichimoku system with Tenkan/Kijun cross and Senkou Span cloud filter, combined with weekly trend filter. Uses Ichimoku for momentum and trend identification, with weekly timeframe to filter counter-trend trades. Designed to work in both bull and bear markets by only taking trades aligned with the weekly trend direction, reducing whipsaws during sideways periods. Targets 12-37 trades/year by requiring multiple confirmations (TK cross, cloud position, weekly trend) for entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # Calculate weekly Ichimoku components for trend filter
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9w = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    low_9w = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    tenkan_sen_w = (high_9w + low_9w) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26w = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    low_26w = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    kijun_sen_w = (high_26w + low_26w) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a_w = (tenkan_sen_w + kijun_sen_w) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52w = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    low_52w = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    senkou_span_b_w = (high_52w + low_52w) / 2
    
    # Align weekly Ichimoku to 6h
    tenkan_sen_w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen_w)
    kijun_sen_w_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen_w)
    senkou_span_a_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a_w)
    senkou_span_b_w_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b_w)
    
    # Calculate 6h Ichimoku for entry signals
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (high_52 + low_52) / 2
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For signal, we use current price vs Senkou Span (cloud)
    
    # Weekly trend filter: price above/both Senkou spans = bullish, below/both = bearish
    # Cloud top = max(Senkou A, Senkou B), Cloud bottom = min(Senkou A, Senkou B)
    cloud_top_w = np.maximum(senkou_span_a_w_aligned, senkou_span_b_w_aligned)
    cloud_bottom_w = np.minimum(senkou_span_a_w_aligned, senkou_span_b_w_aligned)
    
    weekly_bullish = close > cloud_top_w
    weekly_bearish = close < cloud_bottom_w
    
    # 6h Ichimoku signals
    # Tenkan/Kijun cross: bullish when Tenkan crosses above Kijun
    tk_cross_up = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_cross_down = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    
    # Price position relative to cloud
    cloud_top = np.maximum(senkou_span_a, senkou_span_b)
    cloud_bottom = np.minimum(senkou_span_a, senkou_span_b)
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: TK cross up + price above cloud + weekly bullish
        long_entry = tk_cross_up[i] and price_above_cloud[i] and weekly_bullish[i]
        
        # Short: TK cross down + price below cloud + weekly bearish
        short_entry = tk_cross_down[i] and price_below_cloud[i] and weekly_bearish[i]
        
        # Exit when TK cross reverses or price enters cloud
        long_exit = tk_cross_down[i] or (close[i] >= cloud_bottom[i] and close[i] <= cloud_top[i])
        short_exit = tk_cross_up[i] or (close[i] >= cloud_bottom[i] and close[i] <= cloud_top[i])
        
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
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Ichimoku_Cloud_Trend_WeeklyFilter"
timeframe = "6h"
leverage = 1.0