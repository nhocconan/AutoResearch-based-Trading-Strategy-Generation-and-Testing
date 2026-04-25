#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1wTrend_Filter
Hypothesis: Ichimoku Tenkan-Kijun cross on 6h with weekly trend filter (price above/below weekly Kumo cloud) captures strong trend continuations while avoiding counter-trend whipsaws. Weekly cloud acts as dynamic support/resistance. Works in both bull and bear markets by only taking trades in direction of weekly trend. Target: 12-30 trades/year per symbol.
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
    
    # 1w data for weekly trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Weekly trend filter: price vs weekly Kumo cloud
    # Kumo top = max(Senkou Span A, Senkou Span B)
    # Kumo bottom = min(Senkou Span A, Senkou Span B)
    # Note: Ichimoku lines are plotted 26 periods ahead, so for current price we use unshifted Senkou spans
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # Align weekly Kumo to 6h
    kumo_top_aligned = align_htf_to_ltf(prices, df_1w, kumo_top)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1w, kumo_bottom)
    
    # TK cross signals
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))  # Tenkan crosses above Kijun
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))  # Tenkan crosses below Kijun
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after Ichimoku warmup (52 periods for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if weekly data not ready
        if np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TK bullish cross AND price above weekly Kumo (bullish weekly trend)
            if tk_cross_above[i] and close[i] > kumo_top_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TK bearish cross AND price below weekly Kumo (bearish weekly trend)
            elif tk_cross_below[i] and close[i] < kumo_bottom_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold until TK bearish cross OR price drops below weekly Kumo bottom
            signals[i] = 0.25
            if tk_cross_below[i] or close[i] < kumo_bottom_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold until TK bullish cross OR price rises above weekly Kumo top
            signals[i] = -0.25
            if tk_cross_above[i] or close[i] > kumo_top_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0