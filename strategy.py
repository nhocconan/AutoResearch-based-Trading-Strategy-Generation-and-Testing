#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_Filter_1dTrend
Hypothesis: 6h Ichimoku TK cross with 1d Kumo twist filter (Senkou Span A/B cross) and volume confirmation.
Works in bull/bear: Kumo twist indicates regime change, TK cross provides entry timing in new trend direction.
Target: 75-150 trades over 4 years (19-38/year). Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.8x 20-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    # Load 1d data for HTF Ichimoku
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need 52 for Senkou Span B
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
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
    
    # Align all Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Kumo twist: Senkou A crosses Senkou Bullish: Senkou A > Senkou B
    # Bearish: Senkou A < Senkou B
    senkou_a_bullish = senkou_a_aligned > senkou_b_aligned
    senkou_a_bearish = senkou_a_aligned < senkou_b_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 52-period data)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Bullish TK cross: Tenkan crosses above Kijun
        tk_bullish_cross = (i > start_idx and 
                           tenkan_aligned[i-1] <= kijun_aligned[i-1] and 
                           tenkan_aligned[i] > kijun_aligned[i])
        
        # Bearish TK cross: Tenkan crosses below Kijun
        tk_bearish_cross = (i > start_idx and 
                           tenkan_aligned[i-1] >= kijun_aligned[i-1] and 
                           tenkan_aligned[i] < kijun_aligned[i])
        
        # Long logic: bullish TK cross + bullish Kumo (Senkou A > Senkou B) + volume confirm
        if tk_bullish_cross and senkou_a_bullish[i] and volume_confirm[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: bearish TK cross + bearish Kumo (Senkou A < Senkou B) + volume confirm
        elif tk_bearish_cross and senkou_a_bearish[i] and volume_confirm[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: long exits on bearish TK cross OR Kumo turns bearish
        elif position == 1 and (tk_bearish_cross or not senkou_a_bullish[i]):
            signals[i] = 0.0
            position = 0
        # Exit: short exits on bullish TK cross OR Kumo turns bullish
        elif position == -1 and (tk_bullish_cross or not senkou_a_bearish[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_Filter_1dTrend"
timeframe = "6h"
leverage = 1.0