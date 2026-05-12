#!/usr/bin/env python3
# 6h_1D_1W_Ichimoku_Tenkan_Kijun_Cross_Cloud
# Hypothesis: Ichimoku Kijun/Tenkan cross with cloud filter on 1d for trend direction, 
# using 1w to confirm long-term trend alignment. Works in bull/bear by requiring 
# both 1d and 1w trend alignment before taking signals on 6h.
# Targets 12-37 trades/year on 6h timeframe to avoid fee drag.

name = "6h_1D_1W_Ichimoku_Tenkan_Kijun_Cross_Cloud"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components: tenkan, kijun, senkou_a, senkou_b"""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 1d data for Ichimoku (trend direction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    # Get 1w data for long-term trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)

    # Calculate Ichimoku on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Calculate Ichimoku on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w = calculate_ichimoku(high_1w, low_1w, close_1w)

    # Align Ichimoku components to 6h timeframe
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w, additional_delay_bars=0)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w, additional_delay_bars=0)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w, additional_delay_bars=0)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w, additional_delay_bars=0)

    # Determine cloud boundaries (senkou span A/B)
    # For 1d cloud
    cloud_top_1d = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom_1d = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # For 1w cloud
    cloud_top_1w = np.maximum(senkou_a_1w_aligned, senkou_b_1w_aligned)
    cloud_bottom_1w = np.minimum(senkou_a_1w_aligned, senkou_b_1w_aligned)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top_1d[i]) or np.isnan(cloud_bottom_1d[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(cloud_top_1w[i]) or np.isnan(cloud_bottom_1w[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # 1d trend conditions
        bullish_1d = (tenkan_1d_aligned[i] > kijun_1d_aligned[i]) and (close[i] > cloud_top_1d[i])
        bearish_1d = (tenkan_1d_aligned[i] < kijun_1d_aligned[i]) and (close[i] < cloud_bottom_1d[i])
        
        # 1w trend conditions (long-term filter)
        bullish_1w = (tenkan_1w_aligned[i] > kijun_1w_aligned[i]) and (close[i] > cloud_top_1w[i])
        bearish_1w = (tenkan_1w_aligned[i] < kijun_1w_aligned[i]) and (close[i] < cloud_bottom_1w[i])

        if position == 0:
            # LONG: Tenkan/Kijun cross bullish + price above cloud (1d) + 1w bullish alignment
            if bullish_1d and bullish_1w:
                signals[i] = 0.25
                position = 1
            # SHORT: Tenkan/Kijun cross bearish + price below cloud (1d) + 1w bearish alignment
            elif bearish_1d and bearish_1w:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Tenkan/Kijun cross bearish OR price drops below cloud
            if (tenkan_1d_aligned[i] < kijun_1d_aligned[i]) or (close[i] < cloud_top_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Tenkan/Kijun cross bullish OR price rises above cloud
            if (tenkan_1d_aligned[i] > kijun_1d_aligned[i]) or (close[i] > cloud_bottom_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals