#!/usr/bin/env python3
# 6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_Volume
# Hypothesis: On 6-hour chart, use Ichimoku Tenkan/Kijun cross as entry signal with
# 1-day Ichimoku cloud for trend filter and volume spike for confirmation.
# Long: TK cross bullish, price above 1d cloud (bullish trend), volume > 2x 20-period avg.
# Short: TK cross bearish, price below 1d cloud (bearish trend), volume > 2x 20-period avg.
# Exit: TK cross reverses or price crosses Kijun (6h). Designed for ~15-30 trades/year.

name = "6h_Ichimoku_TK_Cross_CloudFilter_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku cloud calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate 6h Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods
    # For cloud, we need current Senkou spans (calculated 26 periods ago)
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_a = ((tenkan + kijun) / 2)  # Will be shifted via alignment logic
    senkou_b = (high_52 + low_52) / 2
    
    # Calculate daily Ichimoku cloud components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Tenkan (9-period) and Kijun (26-period)
    high_9_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    low_9_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (high_9_1d + low_9_1d) / 2
    
    high_26_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    low_26_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (high_26_1d + low_26_1d) / 2
    
    # Daily Senkou Span B (52-period)
    high_52_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    low_52_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = (high_52_1d + low_52_1d) / 2
    
    # Daily Senkou Span A: (Tenkan_1d + Kijun_1d)/2
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    # Align daily Ichimoku components to 6h timeframe (use previous day's values)
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d)
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # Cloud top and bottom (Senkou A and B)
    cloud_top = np.maximum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    cloud_bottom = np.minimum(senkou_a_1d_aligned, senkou_b_1d_aligned)
    
    # Volume spike detection: 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # Need Kijun (26) and volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TK cross signals
        tk_cross_bullish = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_bearish = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        if position == 0:
            # Long: bullish TK cross, price above cloud (bullish trend), volume spike
            if (tk_cross_bullish and 
                close[i] > cloud_top[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish TK cross, price below cloud (bearish trend), volume spike
            elif (tk_cross_bearish and 
                  close[i] < cloud_bottom[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish TK cross OR price crosses below Kijun
            if tk_cross_bearish or close[i] < kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish TK cross OR price crosses above Kijun
            if tk_cross_bullish or close[i] > kijun[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals