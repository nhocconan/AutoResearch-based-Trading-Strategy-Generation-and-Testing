#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d trend filter and volume confirmation
Hypothesis: Ichimoku provides robust trend direction and support/resistance levels.
Price above/below cloud indicates trend, TK cross provides entry signals.
1d trend filter prevents counter-trend trades. Volume confirms institutional participation.
Works in bull (buy above cloud) and bear (sell below cloud) markets. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_prev = np.roll(ema200_1d, 1)
    ema200_1d_prev[0] = ema200_1d[0]
    ema200_rising = ema200_1d > ema200_1d_prev
    ema200_falling = ema200_1d < ema200_1d_prev
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_rising_aligned = align_htf_to_ltf(prices, df_1d, ema200_rising)
    ema200_falling_aligned = align_htf_to_ltf(prices, df_1d, ema200_falling)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used in signals to avoid look-ahead
    
    # Determine cloud (Kumo): upper and lower boundaries
    # For forward projection, we need to shift Senkou spans forward
    # But for current cloud, we use values that were plotted 26 periods ago
    senkou_a_shifted = np.roll(senkou_a, 26)
    senkou_b_shifted = np.roll(senkou_b, 26)
    # Initialize first 26 values
    senkou_a_shifted[:26] = senkou_a[:26]
    senkou_b_shifted[:26] = senkou_b[:26]
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross signals
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need 52 for Senkou B)
    start = 52
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema200_rising_aligned[i]) or 
            np.isnan(ema200_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite TK cross or price crosses opposite cloud boundary
        if position == 1:  # long position
            # Exit: TK cross below OR price falls below cloud bottom
            if tk_cross_below[i] or close[i] <= cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TK cross above OR price rises above cloud top
            if tk_cross_above[i] or close[i] >= cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: TK cross + price relative to cloud + trend + volume
            # Long: TK cross bullish + price above cloud + bullish trend + volume
            long_condition = (tk_cross_above[i] and 
                             close[i] > cloud_top[i] and 
                             ema200_rising_aligned[i] and 
                             volume[i] > vol_ema[i] * 1.5)
            
            # Short: TK cross bearish + price below cloud + bearish trend + volume
            short_condition = (tk_cross_below[i] and 
                              close[i] < cloud_bottom[i] and 
                              ema200_falling_aligned[i] and 
                              volume[i] > vol_ema[i] * 1.5)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals