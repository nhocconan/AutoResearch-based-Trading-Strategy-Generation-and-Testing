#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Cloud_Trend_Filter
Hypothesis: Ichimoku Cloud on daily timeframe provides strong trend direction and support/resistance.
On 6h timeframe, we enter long when Tenkan-sen crosses above Kijun-sen and price is above the cloud,
and short when Tenkan-sen crosses below Kijun-sen and price is below the cloud.
The cloud (Senkou Span A/B) acts as dynamic support/resistance, making this effective in both bull and bear markets.
We add volume confirmation (current volume > 1.5x 20-period average) to filter false signals.
Target: 15-25 trades/year per symbol.
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
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Senkou Span B
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1d).rolling(window=9, min_periods=9).max()
    low_9 = pd.Series(low_1d).rolling(window=9, min_periods=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1d).rolling(window=26, min_periods=26).max()
    low_26 = pd.Series(low_1d).rolling(window=26, min_periods=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1d).rolling(window=52, min_periods=52).max()
    low_52 = pd.Series(low_1d).rolling(window=52, min_periods=52).min()
    senkou_span_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(52, n):  # Start after Senkou Span B is available
        # Skip if any required data is not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # Bullish conditions: Tenkan-sen crosses above Kijun-sen AND price above cloud
        tenkan_cross_above = tenkan_sen_aligned[i] > kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] <= kijun_sen_aligned[i-1]
        price_above_cloud = close[i] > cloud_top
        
        # Bearish conditions: Tenkan-sen crosses below Kijun-sen AND price below cloud
        tenkan_cross_below = tenkan_sen_aligned[i] < kijun_sen_aligned[i] and tenkan_sen_aligned[i-1] >= kijun_sen_aligned[i-1]
        price_below_cloud = close[i] < cloud_bottom
        
        # Long entry: bullish cross + price above cloud + volume expansion
        long_entry = tenkan_cross_above and price_above_cloud and volume_expansion[i]
        
        # Short entry: bearish cross + price below cloud + volume expansion
        short_entry = tenkan_cross_below and price_below_cloud and volume_expansion[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Ichimoku_Cloud_Trend_Filter"
timeframe = "6h"
leverage = 1.0