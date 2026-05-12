#!/usr/bin/env python3
# 6h_Ichimoku_Cloud_Breakout_1dTrend
# Hypothesis: Use Ichimoku cloud from daily timeframe as trend filter and support/resistance, with Tenkan-Kijun cross on 6h for entry.
# In bull markets, price stays above cloud; in bear markets, price stays below cloud.
# Tenkan-Kijun cross provides momentum signals aligned with the daily trend.
# Cloud acts as dynamic support/resistance, reducing false breakouts.
# Designed for 12-37 trades/year per symbol, works in both bull and bear via trend filter.

name = "6h_Ichimoku_Cloud_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: Tenkan, Kijun, Senkou A, Senkou B"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max()
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min()
    senkou_b = (period52_high + period52_low) / 2
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)

    # Calculate Ichimoku components on daily data
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)

    # Determine cloud boundaries (Senkou A and B)
    upper_cloud = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    lower_cloud = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Trend filter: price above cloud = uptrend, below cloud = downtrend
    price_above_cloud = close > upper_cloud
    price_below_cloud = close < lower_cloud

    # Tenkan-Kijun cross on 6h for entry signals
    # Calculate Tenkan and Kijun on 6h data
    period9_high_6h = pd.Series(high).rolling(window=9, min_periods=9).max()
    period9_low_6h = pd.Series(low).rolling(window=9, min_periods=9).min()
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    period26_high_6h = pd.Series(high).rolling(window=26, min_periods=26).max()
    period26_low_6h = pd.Series(low).rolling(window=26, min_periods=26).min()
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Tenkan-Kijun cross signals
    tk_cross_up = (tenkan_6h > kijun_6h) & (tenkan_6h.shift(1) <= kijun_6h.shift(1))
    tk_cross_down = (tenkan_6h < kijun_6h) & (tenkan_6h.shift(1) >= kijun_6h.shift(1))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(52, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i]) or
            np.isnan(tk_cross_up[i]) or np.isnan(tk_cross_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Tenkan crosses above Kijun AND price above daily cloud (uptrend)
            if tk_cross_up[i] and price_above_cloud[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan crosses below Kijun AND price below daily cloud (downtrend)
            elif tk_cross_down[i] and price_below_cloud[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan crosses below Kijun OR price falls below cloud
            if tk_cross_down[i] or not price_above_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan crosses above Kijun OR price rises above cloud
            if tk_cross_up[i] or not price_below_cloud[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals