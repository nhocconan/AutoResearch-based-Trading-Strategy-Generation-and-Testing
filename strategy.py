#!/usr/bin/env python3
"""
6h_12h_Ichimoku_Kumo_Breakout_Trend_v1
Hypothesis: Use 12h Ichimoku cloud (Senkou Span A/B) to determine long-term trend and 6h Tenkan/Kijun cross for entry timing.
Only go long when price breaks above the 12h cloud with bullish TK cross (Tenkan > Kijun), short when breaks below cloud with bearish TK cross.
The cloud acts as dynamic support/resistance, reducing false breakouts. Works in both bull/bear markets as trend filter adapts.
Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.
"""

name = "6h_12h_Ichimoku_Kumo_Breakout_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 52:  # need at least 52 periods for Ichimoku calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Ichimoku calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 52:
        return np.zeros(n)
    
    # Calculate Ichimoku components on 12h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = df_12h['high'].rolling(window=9, min_periods=9).max()
    low_9 = df_12h['low'].rolling(window=9, min_periods=9).min()
    tenkan = ((high_9 + low_9) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = df_12h['high'].rolling(window=26, min_periods=26).max()
    low_26 = df_12h['low'].rolling(window=26, min_periods=26).min()
    kijun = ((high_26 + low_26) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = df_12h['high'].rolling(window=52, min_periods=52).max()
    low_52 = df_12h['low'].rolling(window=52, min_periods=52).min()
    senkou_b = ((high_52 + low_52) / 2).values
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # Not used for signals but needed for cloud calculation
    
    # Align Ichimoku components to 6h timeframe (wait for 12h bar to close)
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # Top of cloud = max(Senkou A, Senkou B)
    # Bottom of cloud = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # TK cross signals
    tk_bullish = tenkan_aligned > kijun_aligned
    tk_bearish = tenkan_aligned < kijun_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):
        if (np.isnan(tenkan_aligned[i]) or
            np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above cloud + bullish TK cross
            if (close[i] > cloud_top[i] and tk_bullish[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below cloud + bearish TK cross
            elif (close[i] < cloud_bottom[i] and tk_bearish[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters cloud OR TK cross turns bearish
            if (close[i] > cloud_bottom[i] and close[i] < cloud_top[i]) or \
               (~tk_bullish[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters cloud OR TK cross turns bullish
            if (close[i] > cloud_bottom[i] and close[i] < cloud_top[i]) or \
               tk_bullish[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals