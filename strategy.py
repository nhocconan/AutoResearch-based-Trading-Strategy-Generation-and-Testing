#!/usr/bin/env python3
"""
4h_1d_Ichimoku_Kumo_Trend_Breakout
Hypothesis: Use Ichimoku Kumo (cloud) from 1d as trend filter and support/resistance, combined with 4h price breakout above/below cloud edges and volume confirmation. The Kumo acts as dynamic support/resistance that adapts to volatility, reducing whipsaws in both bull and bear markets. Enter long when price breaks above Kumo top with volume and Tenkan > Kijun (bullish alignment), short when price breaks below Kumo bottom with volume and Tenkan < Kijun. Exit when price re-enters the cloud. Targets 20-35 trades/year by requiring alignment of trend, breakout, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on 1d
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Kumo cloud: Senkou A and B shifted forward 26 periods
    # But for cloud calculation, we need current Senkou spans (not shifted)
    # The cloud ahead is what matters for support/resistance
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # First 26 values are invalid due to roll
    senkou_a_shifted[:26] = np.nan
    senkou_b_shifted[:26] = np.nan
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    kumo_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Align Ichimoku components to 4h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # need Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Kumo top, with volume, and bullish alignment (Tenkan > Kijun)
            if (close[i] > kumo_top_aligned[i] and vol_confirm[i] and 
                tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Kumo bottom, with volume, and bearish alignment (Tenkan < Kijun)
            elif (close[i] < kumo_bottom_aligned[i] and vol_confirm[i] and 
                  tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price re-enters Kumo (below Kumo top) or Tenkan < Kijun (trend weakening)
            if (close[i] < kumo_top_aligned[i] or 
                tenkan_aligned[i] < kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters Kumo (above Kumo bottom) or Tenkan > Kijun (trend weakening)
            if (close[i] > kumo_bottom_aligned[i] or 
                tenkan_aligned[i] > kijun_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_Ichimoku_Kumo_Trend_Breakout"
timeframe = "4h"
leverage = 1.0