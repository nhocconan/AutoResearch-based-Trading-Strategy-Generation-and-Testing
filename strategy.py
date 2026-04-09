#!/usr/bin/env python3
# 6h_weekly_ichimoku_cloud_breakout_v1
# Hypothesis: 6h strategy using 1w Ichimoku cloud for trend direction and 6h price action for timing.
# Long: Price breaks above weekly Kumo (cloud) with Tenkan > Kijun (bullish TK cross) and close > open
# Short: Price breaks below weekly Kumo with Tenkan < Kijun (bearish TK cross) and close < open
# Exit: Price returns to opposite cloud edge (Tenkan-Kijun midpoint) or TK cross reverses
# Uses 6h primary timeframe with 1w HTF for Ichimoku calculation.
# Ichimoku works in both bull and bear markets by capturing institutional trends with confirmation.
# Target: 75-150 total trades over 4 years (19-38/year) to reduce fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_ichimoku_cloud_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_prices = prices['open'].values
    
    # Get 1w data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 52 weeks for proper calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high + period9_low) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high + period26_low) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (period52_high + period52_low) / 2.0
    
    # Align Ichimoku components to 6h timeframe
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_a_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_a_1w)
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup period for Ichimoku
        # Skip if any required data is NaN
        if (np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or 
            np.isnan(senkou_a_1w_aligned[i]) or np.isnan(senkou_b_1w_aligned[i]) or
            np.isnan(close[i]) or np.isnan(open_prices[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        lower_cloud = np.minimum(senkou_a_1w_aligned[i], senkou_b_1w_aligned[i])
        # Cloud midpoint (Tenkan-Kijun midpoint) for exit
        cloud_midpoint = (senkou_a_1w_aligned[i] + senkou_b_1w_aligned[i]) / 2.0
        
        # TK cross conditions
        bullish_tk = tenkan_1w_aligned[i] > kijun_1w_aligned[i]
        bearish_tk = tenkan_1w_aligned[i] < kijun_1w_aligned[i]
        # Bullish candle: close > open
        bullish_candle = close[i] > open_prices[i]
        # Bearish candle: close < open
        bearish_candle = close[i] < open_prices[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to cloud midpoint OR TK cross turns bearish
            if close[i] <= cloud_midpoint or not bullish_tk:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to cloud midpoint OR TK cross turns bullish
            if close[i] >= cloud_midpoint or not bearish_tk:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above upper cloud with bullish TK cross and bullish candle
            if close[i] > upper_cloud and bullish_tk and bullish_candle:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below lower cloud with bearish TK cross and bearish candle
            elif close[i] < lower_cloud and bearish_tk and bearish_candle:
                position = -1
                signals[i] = -0.25
    
    return signals