#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloudFilter
Hypothesis: On 6h timeframe, enter long when Tenkan-Kijun cross above with price above 1d Ichimoku cloud (bullish bias), enter short when cross below with price below 1d cloud (bearish bias). Uses 1d cloud as higher-timeframe trend filter to avoid counter-trend whipsaws. Designed for low trade frequency (50-150 total over 4 years) with discrete position sizing (0.25) to minimize fee drag. Works in both bull and bear markets by following 1d cloud bias.
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
    
    # Calculate Ichimoku components on 6h timeframe
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2.0
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Load 1d data for Ichimoku cloud (Senkou Span A & B)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen 1d (9-period)
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max().values + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min().values) / 2.0
    
    # Kijun-sen 1d (26-period)
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max().values + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min().values) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_span_a_1d = ((tenkan_1d + kijun_1d) / 2.0)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    senkou_span_b_1d = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max().values + 
                         pd.Series(low_1d).rolling(window=52, min_periods=52).min().values) / 2.0)
    
    # The cloud is between Senkou Span A and B
    # Align to 6h timeframe with 26-period delay (since Senkou spans are plotted ahead)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d, additional_delay_bars=26)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d, additional_delay_bars=26)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    cloud_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Price above/below cloud signals
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 26-period for Kijun + 26 for Senkou delay)
    start_idx = 26 + 26
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: TK cross above + price above 1d cloud
        if tk_cross_above[i] and price_above_cloud[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: TK cross below + price below 1d cloud
        elif tk_cross_below[i] and price_below_cloud[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: TK cross in opposite direction
        elif position == 1 and tk_cross_below[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and tk_cross_above[i]:
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

name = "6h_Ichimoku_TK_Cross_1dCloudFilter"
timeframe = "6h"
leverage = 1.0