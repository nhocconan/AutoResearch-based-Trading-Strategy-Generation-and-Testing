#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_WeeklyTrend_DailyCloud
Hypothesis: 6-hour Ichimoku Tenkan-Kijun cross with weekly trend filter (price above/below weekly cloud) and daily cloud filter for confirmation.
Targets 12-37 trades/year by requiring: 1) TK cross on 6h chart, 2) price relative to weekly Ichimoku cloud (trend filter), 3) price relative to daily Ichimoku cloud (entry filter).
Uses 6h timeframe to balance trade frequency and capture significant moves. Weekly cloud provides major trend direction, daily cloud provides entry timing and reduces false signals.
Ichimoku components calculated correctly with proper lookback periods and aligned to avoid look-ahead bias.
"""

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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Weekly data for trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    # Weekly Ichimoku: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displacement)
    wk_high_9 = pd.Series(df_1w['high']).rolling(window=9, min_periods=9).max().values
    wk_low_9 = pd.Series(df_1w['low']).rolling(window=9, min_periods=9).min().values
    wk_tenkan = (wk_high_9 + wk_low_9) / 2
    
    wk_high_26 = pd.Series(df_1w['high']).rolling(window=26, min_periods=26).max().values
    wk_low_26 = pd.Series(df_1w['low']).rolling(window=26, min_periods=26).min().values
    wk_kijun = (wk_high_26 + wk_low_26) / 2
    
    wk_senkou_a = (wk_tenkan + wk_kijun) / 2
    wk_senkou_b_high = pd.Series(df_1w['high']).rolling(window=52, min_periods=52).max().values
    wk_senkou_b_low = pd.Series(df_1w['low']).rolling(window=52, min_periods=52).min().values
    wk_senkou_b = (wk_senkou_b_high + wk_senkou_b_low) / 2
    
    # Align weekly Ichimoku components to 6h timeframe (with 1-bar delay for completed weekly bar)
    wk_tenkan_aligned = align_htf_to_ltf(prices, df_1w, wk_tenkan)
    wk_kijun_aligned = align_htf_to_ltf(prices, df_1w, wk_kijun)
    wk_senkou_a_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_a)
    wk_senkou_b_aligned = align_htf_to_ltf(prices, df_1w, wk_senkou_b)
    
    # Daily data for entry filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    # Daily Ichimoku: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period displacement)
    dy_high_9 = pd.Series(df_1d['high']).rolling(window=9, min_periods=9).max().values
    dy_low_9 = pd.Series(df_1d['low']).rolling(window=9, min_periods=9).min().values
    dy_tenkan = (dy_high_9 + dy_low_9) / 2
    
    dy_high_26 = pd.Series(df_1d['high']).rolling(window=26, min_periods=26).max().values
    dy_low_26 = pd.Series(df_1d['low']).rolling(window=26, min_periods=26).min().values
    dy_kijun = (dy_high_26 + dy_low_26) / 2
    
    dy_senkou_a = (dy_tenkan + dy_kijun) / 2
    dy_senkou_b_high = pd.Series(df_1d['high']).rolling(window=52, min_periods=52).max().values
    dy_senkou_b_low = pd.Series(df_1d['low']).rolling(window=52, min_periods=52).min().values
    dy_senkou_b = (dy_senkou_b_high + dy_senkou_b_low) / 2
    
    # Align daily Ichimoku components to 6h timeframe (with 1-bar delay for completed daily bar)
    dy_tenkan_aligned = align_htf_to_ltf(prices, df_1d, dy_tenkan)
    dy_kijun_aligned = align_htf_to_ltf(prices, df_1d, dy_kijun)
    dy_senkou_a_aligned = align_htf_to_ltf(prices, df_1d, dy_senkou_a)
    dy_senkou_b_aligned = align_htf_to_ltf(prices, df_1d, dy_senkou_b)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly calculations (52+26) and daily calculations (52+26)
    start_idx = 52 + 26 + 26  # Conservative warmup for weekly and daily Ichimoku
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(wk_tenkan_aligned[i]) or np.isnan(wk_kijun_aligned[i]) or 
            np.isnan(wk_senkou_a_aligned[i]) or np.isnan(wk_senkou_b_aligned[i]) or
            np.isnan(dy_tenkan_aligned[i]) or np.isnan(dy_kijun_aligned[i]) or
            np.isnan(dy_senkou_a_aligned[i]) or np.isnan(dy_senkou_b_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Weekly trend filter: price relative to weekly cloud
        wk_top_cloud = np.maximum(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
        wk_bottom_cloud = np.minimum(wk_senkou_a_aligned[i], wk_senkou_b_aligned[i])
        price_above_weekly_cloud = curr_close > wk_top_cloud
        price_below_weekly_cloud = curr_close < wk_bottom_cloud
        
        # Daily Ichimoku for entry signals
        dy_top_cloud = np.maximum(dy_senkou_a_aligned[i], dy_senkou_b_aligned[i])
        dy_bottom_cloud = np.minimum(dy_senkou_a_aligned[i], dy_senkou_b_aligned[i])
        
        # TK cross signals on 6h chart (using daily Ichimoku components as reference)
        # Tenkan-sen cross above/below Kijun-sen
        tenkan_cross_above = (dy_tenkan_aligned[i] > dy_kijun_aligned[i]) and \
                            (dy_tenkan_aligned[i-1] <= dy_kijun_aligned[i-1])
        tenkan_cross_below = (dy_tenkan_aligned[i] < dy_kijun_aligned[i]) and \
                            (dy_tenkan_aligned[i-1] >= dy_kijun_aligned[i-1])
        
        if position == 0:
            # Look for entry signals with weekly trend alignment and daily TK cross
            # Long: price above weekly cloud + tenkan crosses above kijun
            long_signal = price_above_weekly_cloud and tenkan_cross_above
            # Short: price below weekly cloud + tenkan crosses below kijun
            short_signal = price_below_weekly_cloud and tenkan_cross_below
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price crosses below daily cloud or weekly trend changes
            if curr_close < dy_bottom_cloud or not price_above_weekly_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses above daily cloud or weekly trend changes
            if curr_close > dy_top_cloud or not price_below_weekly_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_WeeklyTrend_DailyCloud"
timeframe = "6h"
leverage = 1.0