#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_v1
Strategy: Ichimoku cloud breakout with daily trend filter.
Long: Price breaks above Kumo (cloud) with Tenkan > Kijun in bullish daily trend.
Short: Price breaks below Kumo with Tenkan < Kijun in bearish daily trend.
Designed for 6h timeframe: ~15-30 trades/year per symbol (60-120 total over 4 years).
Works in bull/bear via daily trend filter and cloud breakout confirmation.
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
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    senkou_span_b = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                     pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    
    # Kumo (cloud) boundaries: Senkou Span A and B shifted forward 26 periods
    # For backtesting, we need the cloud values from 26 periods ago
    senkou_span_a_shifted = senkou_span_a.shift(26)
    senkou_span_b_shifted = senkou_span_b.shift(26)
    
    # Daily trend filter: price above/below 200 EMA
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean()
    
    # Align all daily data to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_shifted.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_shifted.values)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for Ichimoku calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Cloud boundaries (top and bottom of Kumo)
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan_aligned[i] > kijun_aligned[i]
        tenkan_below_kijun = tenkan_aligned[i] < kijun_aligned[i]
        
        # Price relative to cloud
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Daily trend filter
        daily_uptrend = close[i] > ema_200_aligned[i]
        daily_downtrend = close[i] < ema_200_aligned[i]
        
        if position == 0:
            # Long: price breaks above cloud + Tenkan > Kijun + daily uptrend
            if price_above_cloud and tenkan_above_kijun and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below cloud + Tenkan < Kijun + daily downtrend
            elif price_below_cloud and tenkan_below_kijun and daily_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below cloud or Tenkan < Kijun
            if price_below_cloud or tenkan_below_kijun:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above cloud or Tenkan > Kijun
            if price_above_cloud or tenkan_above_kijun:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_v1"
timeframe = "6h"
leverage = 1.0