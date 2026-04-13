#!/usr/bin/env python3
"""
6h_1w_1d_Ichimoku_Trend_Follow_v2
Hypothesis: Ichimoku system on weekly timeframe for trend direction + 1d cloud filter + 6s entry timing.
Weekly Tenkan/Kijun cross gives major trend, price above/below weekly cloud filters false signals.
1d Tenkan/Kijun cross provides entry timing with trend alignment. Works in bull (buy above weekly cloud)
and bear (sell below weekly cloud). Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A/B, Chikou."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = pd.Series(close).shift(26)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values, chikou.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for main trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Ichimoku
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Get daily data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Ichimoku for entry signals
    tenkan_1d, kijun_1d, _, _, _ = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Align weekly components to 6h
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    # Align daily components to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    
    # Weekly trend: Tenkan > Kijun = bullish, Tenkan < Kijun = bearish
    weekly_bullish = tenkan_1w_aligned > kijun_1w_aligned
    weekly_bearish = tenkan_1w_aligned < kijun_1w_aligned
    
    # Price relative to weekly cloud
    weekly_top_cloud = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    weekly_bottom_cloud = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    price_above_weekly_cloud = close > weekly_top_cloud
    price_below_weekly_cloud = close < weekly_bottom_cloud
    
    # Daily entry signals: Tenkan/Kijun cross
    tenkan_crosses_above_kijun = (tenkan_1d_aligned > kijun_1d_aligned) & (np.roll(tenkan_1d_aligned, 1) <= np.roll(kijun_1d_aligned, 1))
    tenkan_crosses_below_kijun = (tenkan_1d_aligned < kijun_1d_aligned) & (np.roll(tenkan_1d_aligned, 1) >= np.roll(kijun_1d_aligned, 1))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(52, n):
        # Skip if any required data is not ready
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(weekly_top_cloud[i]) or np.isnan(weekly_bottom_cloud[i]) or
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: weekly bullish + price above weekly cloud + daily Tenkan crosses above Kijun
        long_condition = weekly_bullish[i] and price_above_weekly_cloud[i] and tenkan_crosses_above_kijun[i]
        
        # Short: weekly bearish + price below weekly cloud + daily Tenkan crosses below Kijun
        short_condition = weekly_bearish[i] and price_below_weekly_cloud[i] and tenkan_crosses_below_kijun[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1w_1d_Ichimoku_Trend_Follow_v2"
timeframe = "6h"
leverage = 1.0