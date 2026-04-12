#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_ichimoku_trend_v1
# Uses daily Ichimoku Cloud (Tenkan-sen, Kijun-sen, Senkou Span A/B) with 6h price action.
# Long when price > cloud AND Tenkan > Kijun (bullish alignment).
# Short when price < cloud AND Tenkan < Kijun (bearish alignment).
# Cloud acts as dynamic support/resistance, filtering false breakouts.
# Works in bull markets (trend-following above cloud) and bear markets (trend-following below cloud).
# Target: 15-30 trades/year per symbol for low friction and high edge.

name = "6h_1d_ichimoku_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # Ichimoku components (9, 26, 52 periods)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_6h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_6h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Determine cloud boundaries (Senkou Span A/B)
    upper_cloud = np.maximum(senkou_a_6h, senkou_b_6h)
    lower_cloud = np.minimum(senkou_a_6h, senkou_b_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # start after Ichimoku warmup
        # Skip if Ichimoku not ready
        if (np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i]) or 
            np.isnan(upper_cloud[i]) or np.isnan(lower_cloud[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        price_above_cloud = close[i] > upper_cloud[i]
        price_below_cloud = close[i] < lower_cloud[i]
        tenkan_above_kijun = tenkan_6h[i] > kijun_6h[i]
        tenkan_below_kijun = tenkan_6h[i] < kijun_6h[i]
        
        # Long: price above cloud AND Tenkan > Kijun
        if price_above_cloud and tenkan_above_kijun and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price below cloud AND Tenkan < Kijun
        elif price_below_cloud and tenkan_below_kijun and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses into cloud or Tenkan/Kijun cross reverses
        elif ((close[i] >= lower_cloud[i] and close[i] <= upper_cloud[i]) or
              (position == 1 and tenkan_below_kijun) or
              (position == -1 and tenkan_above_kijun)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals