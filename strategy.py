#!/usr/bin/env python3
# 6h_ichimoku_cloud_volume_v1
# Hypothesis: 6h strategy using Ichimoku cloud from 1d for trend direction and 6h TK cross for entry timing.
# Long: Price above 1d Ichimoku cloud, TK line crosses above Kijun line on 6h, volume > 1.5x 20-period average.
# Short: Price below 1d Ichimoku cloud, TK line crosses below Kijun line on 6h, volume > 1.5x 20-period average.
# Exit: Price crosses opposite TK/Kijun line or volume drops below average.
# Uses Ichimoku cloud (Senkou Span A/B) from 1d for robust trend filtering that works in both bull and bear markets.
# TK cross provides timely entries while cloud filter prevents counter-trend trades.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_cloud_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    displacement = 26
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high_1d).rolling(window=tenkan_period, min_periods=tenkan_period).max()
    low_9 = pd.Series(low_1d).rolling(window=tenkan_period, min_periods=tenkan_period).min()
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high_1d).rolling(window=kijun_period, min_periods=kijun_period).max()
    low_26 = pd.Series(low_1d).rolling(window=kijun_period, min_periods=kijun_period).min()
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    high_52 = pd.Series(high_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).max()
    low_52 = pd.Series(low_1d).rolling(window=senkou_span_b_period, min_periods=senkou_span_b_period).min()
    senkou_b = (high_52 + low_52) / 2.0
    
    # Align Ichimoku components to 6h
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan.values)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun.values)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        # Determine if price is above or below cloud
        cloud_top = max(senkou_a_aligned[i], senkou_b_aligned[i])
        cloud_bottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        price_above_cloud = close[i] > cloud_top
        price_below_cloud = close[i] < cloud_bottom
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # TK cross signals
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        if position == 1:  # Long position
            # Exit: Price crosses below Kijun or volume drops
            if (tenkan_aligned[i] < kijun_aligned[i]) or (not volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above Kijun or volume drops
            if (tenkan_aligned[i] > kijun_aligned[i]) or (not volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price above cloud, TK crosses above Kijun, volume confirmed
            if (price_above_cloud and tk_cross_above and volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below cloud, TK crosses below Kijun, volume confirmed
            elif (price_below_cloud and tk_cross_below and volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals