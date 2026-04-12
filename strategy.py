#!/usr/bin/env python3
"""
6h_1d_Ichimoku_Kumo_Twist
Hypothesis: Use Ichimoku Cloud twist (Senkou Span A/B crossover) from daily timeframe as trend filter,
combined with Tenkan/Kijun cross on 6h for entry. Works in bull/bear because cloud twist
indicates trend acceleration, and Tenkan/Kijun provides timely entries with trend alignment.
Targets 60-120 total trades over 4 years (15-30/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Ichimoku_Kumo_Twist"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D DATA FOR ICHIMOKU CLOUD ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan_sen = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                  pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun_sen = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                 pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                      pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2)
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b.values)
    
    # Kumo twist: Senkou Span A crossing above/below Senkou Span B
    # Bullish twist: Span A > Span B (after previously being below)
    # Bearish twist: Span A < Span B (after previously being above)
    span_a_above_b = senkou_span_a_aligned > senkou_span_b_aligned
    span_a_above_b_prev = np.roll(span_a_above_b, 1)
    span_a_above_b_prev[0] = False
    
    # Kumo twist signals (true at the bar where cross occurs)
    bullish_twist = span_a_above_b & (~span_a_above_b_prev)
    bearish_twist = (~span_a_above_b) & span_a_above_b_prev
    
    # Align twist signals to 6b
    bullish_twist_aligned = align_htf_to_ltf(prices, df_1d, bullish_twist.astype(float))
    bearish_twist_aligned = align_htf_to_ltf(prices, df_1d, bearish_twist.astype(float))
    
    # === 6H DATA FOR ENTRY SIGNALS ===
    # Tenkan/Kijun cross on 6h for entry timing
    high_6h = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_6h = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (high_6h + low_6h) / 2
    
    high_6h_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_6h_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_6h = (high_6h_26 + low_6h_26) / 2
    
    # Tenkan/Kijun cross signals
    tenkan_above_kijun = tenkan_6h > kijun_6h
    tenkan_above_kijun_prev = np.roll(tenkan_above_kijun, 1)
    tenkan_above_kijun_prev[0] = False
    
    tk_bullish_cross = tenkan_above_kijun & (~tenkan_above_kijun_prev)
    tk_bearish_cross = (~tenkan_above_kijun) & tenkan_above_kijun_prev
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):
        # Skip if not ready
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(tenkan_6h[i]) or np.isnan(kijun_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current trend from Kumo twist (need sustained signal)
        # We consider trend bullish if we've had a bullish twist more recently than bearish
        # Simplified: use current Span A/B position as trend filter
        bullish_trend = senkou_span_a_aligned[i] > senkou_span_b_aligned[i]
        bearish_trend = senkou_span_a_aligned[i] < senkou_span_b_aligned[i]
        
        # Entry: Tenkan/Kijun cross in direction of Kumo twist trend
        long_entry = tk_bullish_cross[i] and bullish_trend
        short_entry = tk_bearish_cross[i] and bearish_trend
        
        # Exit: opposite Tenkan/Kijun cross or trend change
        exit_long = position == 1 and (tk_bearish_cross[i] or not bullish_trend)
        exit_short = position == -1 and (tk_bullish_cross[i] or not bearish_trend)
        
        # Execute trades
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals