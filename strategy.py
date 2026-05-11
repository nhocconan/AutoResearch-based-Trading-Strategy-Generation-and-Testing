#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Twist
Hypothesis: Ichimoku cloud twist (Senkou Span A/B crossing) on 1d timeframe signals regime change. 
On 6h, enter long when price is above cloud and Tenkan crosses above Kijun; short when below cloud and Tenkan crosses below Kijun.
Use cloud twist as trend filter to avoid whipsaws. Works in both bull (trend following) and bear (counter-trend at extremes) regimes.
Target: 50-150 total trades over 4 years.
"""

name = "6h_Ichimoku_Cloud_Twist"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for Ichimoku calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # --- Ichimoku on 1d ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 ahead
    senkou_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                 pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    # Chikou Span (Lagging Span): close shifted 26 behind (not used for signals)
    
    # Align Ichimoku components to 6h (cloud twist = Senkou A/B cross)
    tenkan_sen_6h = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_6h = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_a_6h = align_htf_to_ltf(prices, df_1d, senkou_a.values)
    senkou_b_6h = align_htf_to_ltf(prices, df_1d, senkou_b.values)
    
    # Cloud twist: Senkou A crosses above/below Senkou B
    # Bullish twist: Senkou A > Senkou B (uptrend)
    # Bearish twist: Senkou A < Senkou B (downtrend)
    bullish_twist = senkou_a_6h > senkou_b_6h
    bearish_twist = senkou_a_6h < senkou_b_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup (max lookback: 52 + 26 shift)
    start_idx = 78
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(tenkan_sen_6h[i]) or np.isnan(kijun_sen_6h[i]) or 
            np.isnan(senkou_a_6h[i]) or np.isnan(senkou_b_6h[i])):
            if position != 0:
                # Exit if cloud twist reverses
                if position == 1 and bearish_twist[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and bullish_twist[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        if position == 0:
            # Look for entries aligned with cloud twist
            # Price above cloud: long bias
            # Price below cloud: short bias
            above_cloud = close_6h[i] > max(senkou_a_6h[i], senkou_b_6h[i])
            below_cloud = close_6h[i] < min(senkou_a_6h[i], senkou_b_6h[i])
            
            # Tenkan/Kijun cross
            tk_cross_up = tenkan_sen_6h[i] > kijun_sen_6h[i] and tenkan_sen_6h[i-1] <= kijun_sen_6h[i-1]
            tk_cross_down = tenkan_sen_6h[i] < kijun_sen_6h[i] and tenkan_sen_6h[i-1] >= kijun_sen_6h[i-1]
            
            if bullish_twist[i] and above_cloud and tk_cross_up:
                signals[i] = 0.25  # long
                position = 1
                entry_price = close_6h[i]
            elif bearish_twist[i] and below_cloud and tk_cross_down:
                signals[i] = -0.25  # short
                position = -1
                entry_price = close_6h[i]
        else:
            # Manage existing position
            if position == 1:
                # Exit conditions for long
                # 1. Cloud twist turns bearish
                # 2. Price falls below cloud
                # 3. Tenkan crosses below Kijun (momentum loss)
                if bearish_twist[i] or not above_cloud or tk_cross_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit conditions for short
                # 1. Cloud twist turns bullish
                # 2. Price rises above cloud
                # 3. Tenkan crosses above Kijun (momentum loss)
                if bullish_twist[i] or not below_cloud or tk_cross_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals