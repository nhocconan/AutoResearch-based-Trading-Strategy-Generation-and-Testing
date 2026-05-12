#!/usr/bin/env python3
name = "4h_Ichimoku_Cloud_Breakout_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku components on daily data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou = close_1d  # Will be aligned later
    
    # Align Ichimoku components to 4h timeframe
    tenkan_4h = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_4h = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_4h = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_4h = align_htf_to_ltf(prices, df_1d, senkou_b)
    chikou_4h = align_htf_to_ltf(prices, df_1d, chikou)
    
    # Calculate 1d EMA50 for additional trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection (15-period)
    vol_ma = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 15)  # Need enough data for Ichimoku and volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan_4h[i]) or 
            np.isnan(kijun_4h[i]) or
            np.isnan(senkou_a_4h[i]) or
            np.isnan(senkou_b_4h[i]) or
            np.isnan(chikou_4h[i]) or
            np.isnan(ema50_1d_4h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine cloud top and bottom
        cloud_top = np.maximum(senkou_a_4h[i], senkou_b_4h[i])
        cloud_bottom = np.minimum(senkou_a_4h[i], senkou_b_4h[i])
        
        if position == 0:
            # Long: Price above cloud + Tenkan > Kijun + price > EMA50 + volume spike
            if (close[i] > cloud_top and 
                tenkan_4h[i] > kijun_4h[i] and
                close[i] > ema50_1d_4h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below cloud + Tenkan < Kijun + price < EMA50 + volume spike
            elif (close[i] < cloud_bottom and 
                  tenkan_4h[i] < kijun_4h[i] and
                  close[i] < ema50_1d_4h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price falls below cloud or Tenkan < Kijun
            if close[i] < cloud_bottom or tenkan_4h[i] < kijun_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price rises above cloud or Tenkan > Kijun
            if close[i] > cloud_top or tenkan_4h[i] > kijun_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals